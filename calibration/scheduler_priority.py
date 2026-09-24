"""Best-effort normal scheduler priority for display and camera workers."""

import os

CALIBRATION_NICE = -10


class CalibrationSchedulerPriority:
    def __init__(self, role: str, requested_nice: int = CALIBRATION_NICE):
        self.role = str(role)
        self.requested_nice = int(requested_nice)
        self.original_nice = self.effective_nice = None
        self.applied = False
        self.error = None
        self._changed = False
        self._attempted = False

    @staticmethod
    def _supported() -> bool:
        return all(hasattr(os, name) for name in ("getpriority", "setpriority", "PRIO_PROCESS"))

    def enable(self) -> dict:
        if self._attempted:
            return self.status()
        self._attempted = True
        if not self._supported():
            self.error = "nice-based priority is unavailable on this platform"
            self._report_failure()
            return self.status()
        try:
            self.original_nice = os.getpriority(os.PRIO_PROCESS, 0)
            target = min(self.original_nice, self.requested_nice)
            if target != self.original_nice:
                os.setpriority(os.PRIO_PROCESS, 0, target)
                self._changed = True
            self.effective_nice = os.getpriority(os.PRIO_PROCESS, 0)
            self.applied = self.effective_nice <= self.requested_nice
        except (OSError, ValueError) as error:
            self.error = str(error).strip() or type(error).__name__
            try:
                self.effective_nice = os.getpriority(os.PRIO_PROCESS, 0)
            except (OSError, ValueError):
                pass
        if self.applied:
            print(f"[CALIBRATION][SCHEDULER] {self.role} nice={self.effective_nice}", flush=True)
        else:
            self._report_failure()
        return self.status()

    def restore(self) -> dict:
        if self._changed and self.original_nice is not None and self._supported():
            try:
                os.setpriority(os.PRIO_PROCESS, 0, self.original_nice)
                self.effective_nice = os.getpriority(os.PRIO_PROCESS, 0)
            except (OSError, ValueError) as error:
                self.error = str(error).strip() or type(error).__name__
                print(f"[WARNING][CALIBRATION][SCHEDULER] Could not restore {self.role}: {self.error}", flush=True)
        result = self.status()
        self._changed = self.applied = self._attempted = False
        self.original_nice = self.effective_nice = self.error = None
        return result

    def status(self) -> dict:
        return {"mechanism": "nice", "requested_nice": self.requested_nice,
                "original_nice": self.original_nice, "effective_nice": self.effective_nice,
                "applied": self.applied, "error": self.error}

    def _report_failure(self) -> None:
        detail = f": {self.error}" if self.error else ""
        print(f"[WARNING][CALIBRATION][SCHEDULER] Could not raise {self.role} to nice "
              f"{self.requested_nice}{detail}; continuing at nice {self.effective_nice}.", flush=True)
