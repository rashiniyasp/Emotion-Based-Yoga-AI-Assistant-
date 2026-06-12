"""
ACORN Live — ACORN v3 Correction (runs in background thread)
Exact same optimization as the notebook, wrapped for threaded use.
"""
import numpy as np
import torch
import torch.nn.functional as F
import threading

from config import (
    CONNECTIONS, ANGLE_TRIPLETS, BODY_SEGMENTS,
    ACORN_NUM_STEPS, ACORN_LR,
    ACORN_LAMBDA_PRED, ACORN_LAMBDA_STICK, ACORN_LAMBDA_LAND,
    ACORN_LAMBDA_ANG, ACORN_LAMBDA_PROX,
)


class Corrector:
    """
    Runs ACORN v3 optimization in a background thread.
    Uses the JCAT model from the Classifier instance.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.model  = classifier.model
        self.device = classifier.device

        # Threading state
        self._lock        = threading.Lock()
        self._thread      = None
        self._result      = None   # (corrected_norm, attn_weights, moved_joints)
        self._is_running  = False

    @property
    def is_running(self):
        return self._is_running

    @property
    def has_result(self):
        with self._lock:
            return self._result is not None

    def get_result(self):
        """Retrieve and consume the result. Returns None if not ready."""
        with self._lock:
            r = self._result
            self._result = None
            return r

    def start_correction(self, incorrect_norm, class_idx, exemplar_norm):
        """
        Launch ACORN v3 optimization in a background thread.
        Non-blocking — returns immediately.
        """
        if self._is_running:
            return  # Already running, skip

        self._is_running = True
        self._thread = threading.Thread(
            target=self._run_optimization,
            args=(incorrect_norm.copy(), class_idx, exemplar_norm.copy()),
            daemon=True
        )
        self._thread.start()

    def _run_optimization(self, incorrect_lm, target_label, exemplar_lm):
        """ACORN v3 optimization — runs in worker thread."""
        try:
            corrected, attn_final = self._acorn_v3(
                incorrect_lm, target_label, exemplar_lm
            )

            # Determine which joints moved significantly
            displacement = np.linalg.norm(corrected - incorrect_lm, axis=1)
            moved_joints = [i for i in range(12) if displacement[i] > 0.01]

            with self._lock:
                self._result = (corrected, attn_final, moved_joints)
        except Exception as e:
            print(f"[Corrector] ACORN failed: {e}")
            with self._lock:
                self._result = None
        finally:
            self._is_running = False

    def _acorn_v3(self, incorrect_lm, target_label, exemplar_lm):
        """Exact ACORN v3 optimization from the notebook."""
        device = self.device
        model  = self.model

        xp = torch.tensor(incorrect_lm, dtype=torch.float32, device=device)
        xe = torch.tensor(exemplar_lm,  dtype=torch.float32, device=device)
        xs = xp.clone().detach().requires_grad_(True)
        tt = torch.tensor([target_label], dtype=torch.long, device=device)
        bones = {(i, j): torch.norm(xp[i] - xp[j]).detach() for i, j in CONNECTIONS}

        opt = torch.optim.Adam([xs], lr=ACORN_LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ACORN_NUM_STEPS)

        for _ in range(ACORN_NUM_STEPS):
            opt.zero_grad()
            attn = self._get_attn(xs.detach())
            logits = model(xs.unsqueeze(0))
            ce = F.cross_entropy(logits, tt)
            if torch.isnan(ce):
                break

            xal = self._safe_procrustes(xe, xs.detach())
            prox = (attn * 12.0) * ((xs - xp.detach()) ** 2).sum(1)

            loss = (ACORN_LAMBDA_PRED  * ce +
                    ACORN_LAMBDA_STICK * self._bone_cost(xs, bones) +
                    ACORN_LAMBDA_LAND  * ((xs - xal) ** 2).mean() +
                    ACORN_LAMBDA_ANG   * self._ang_cost(xs, xal) +
                    ACORN_LAMBDA_PROX  * prox.mean())

            loss.backward()
            if xs.grad is not None:
                xs.grad.data *= (attn * 12.0).unsqueeze(-1)
            torch.nn.utils.clip_grad_norm_([xs], 0.3)
            opt.step()
            sch.step()

        x_opt = xs.detach().cpu().numpy()
        if np.isnan(x_opt).any():
            return incorrect_lm.copy(), None

        # Get final attention for segment refinement
        with torch.no_grad():
            attn_final = self._get_attn(
                torch.tensor(x_opt, dtype=torch.float32, device=device)
            ).cpu().numpy()

        # Apply segment refinement
        x_final = self._apply_segment_refinement(x_opt, incorrect_lm, attn_final)

        return x_final, attn_final

    def _get_attn(self, x, T=0.15):
        with torch.no_grad():
            am = self.model.get_attention_maps(x.unsqueeze(0))
            cs = am[-1].mean(1).squeeze(0)[1:, 1:].sum(0)
            return F.softmax(cs / T, dim=0)

    def _safe_procrustes(self, source, target):
        sc = source.mean(0, keepdim=True)
        tc = target.mean(0, keepdim=True)
        sn = source - sc
        tn = target - tc
        ss = torch.sqrt((sn ** 2).sum() + 1e-6)
        ts = torch.sqrt((tn ** 2).sum() + 1e-6)
        return (sn * torch.clamp(ts / ss, 0.5, 2.0)) + tc

    def _bone_cost(self, x, bones):
        return sum((torch.norm(x[i] - x[j]) - v) ** 2
                    for (i, j), v in bones.items()) / len(bones)

    def _ang_cost(self, x, ex):
        t, c = 0.0, 0
        for ji, (p1, p2, p3) in ANGLE_TRIPLETS.items():
            def a(pts, _p1=p1, _p2=p2, _p3=p3):
                v1 = pts[_p1] - pts[_p2]
                v2 = pts[_p3] - pts[_p2]
                if torch.norm(v1) < 1e-5 or torch.norm(v2) < 1e-5:
                    return None
                return torch.acos(torch.clamp(
                    v1 @ v2 / (torch.norm(v1) * torch.norm(v2) + 1e-6), -1, 1))
            a1, a2 = a(x), a(ex)
            if a1 is not None and a2 is not None:
                t += (a1 - a2) ** 2
                c += 1
        return t / max(c, 1)

    def _apply_segment_refinement(self, x_opt, x_incorrect, attn_weights):
        seg_devs = {}
        for seg_name, idxs in BODY_SEGMENTS.items():
            raw_dev = np.abs(x_opt[idxs] - x_incorrect[idxs]).sum()
            if attn_weights is not None:
                attn_w = float(np.mean([attn_weights[i] for i in idxs]))
                seg_devs[seg_name] = raw_dev * (1.0 + attn_w * 5.0)
            else:
                seg_devs[seg_name] = raw_dev
        max_seg = max(seg_devs, key=seg_devs.get)
        x_final = x_incorrect.copy()
        x_final[BODY_SEGMENTS[max_seg]] = x_opt[BODY_SEGMENTS[max_seg]]
        return x_final
