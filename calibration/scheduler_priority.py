"""Best-effort scheduler priority for timing-sensitive calibration work."""

from __future__ import annotations

import os


CALIBRATION_NICE = -10


class CalibrationSchedulerPriority:
    """Temporarily request a higher normal-scheduler CPU priority.

    A negative nice value is intentionally used instead of a real-time policy:
    it improves preference under load without allowing the calibration loop to
    starve the desktop. Linux normally requires CAP_SYS_NICE (or an equivalent
    limit) to grant the request, so failure is reported but is not fatal.
    """

    def __init__(self, role: str, requested_nice: int = CALIBRATION_NICE):
        self.role = str(role)
        self.requested_nice = int(requested_nice)
        self.original_nice: int | None = None
        self.effective_nice: int | None = None
        self.applied = False
        self.error: str | None = None
        self._changed = False
        self._attempted = False

    @staticmethod
    def _supported() -> bool:
        return all(
            hasattr(os, name)
            for name in ("getpriority", "setpriority", "PRIO_PROCESS")
        )

    def enable(self) -> dict:
        if self._attempted:
            return self.status()
        self._attempted = True
        if not self._supported():
            self.error = "nice-based process priority is unavailable on this platform"
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
            print(
                f"[CALIBRATION][SCHEDULER] {self.role} is using nice "
                f"{self.effective_nice} (requested {self.requested_nice}).",
                flush=True,
            )
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
                print(
                    f"[WARNING][CALIBRATION][SCHEDULER] Could not restore "
                    f"{self.role} to nice {self.original_nice}: {self.error}",
                    flush=True,
                )
        self._changed = False
        self.applied = False
        restored = self.status()
        self.original_nice = None
        self.effective_nice = None
        self.error = None
        self._attempted = False
        return restored

    def status(self) -> dict:
        return {
            "mechanism": "nice",
            "requested_nice": self.requested_nice,
            "original_nice": self.original_nice,
            "effective_nice": self.effective_nice,
            "applied": self.applied,
            "error": self.error,
        }

    def _report_failure(self) -> None:
        detail = f": {self.error}" if self.error else ""
        print(
            f"[WARNING][CALIBRATION][SCHEDULER] Could not raise {self.role} "
            f"to nice {self.requested_nice}{detail}. Continuing at nice "
            f"{self.effective_nice}; grant CAP_SYS_NICE or an equivalent "
            "permission to enable the requested priority.",
            flush=True,
        )
