"""USB transport: vendor control commands, bulk writes and asynchronous bulk reads.

`UsbTransport` talks to real hardware through python-libusb1 (libusb-1.0). On Windows
the libusb1 wheel ships libusb-1.0.dll and works with the WinUSB driver that the
DSView installer puts on the analyser.  Tests substitute a fake with the same methods.
"""
from __future__ import annotations

import struct
import time
from collections.abc import Callable

from . import constants as C


class DSLogicError(RuntimeError):
    pass


class DeviceNotFound(DSLogicError):
    pass


class ReadRequest:
    """One queued bulk-IN transfer. Its callback runs inside `handle_events()` with
    status "ok", "timeout", "cancelled" or "error". `done` turns True once retired."""
    done: bool = False

    def cancel(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class UsbTransport:
    """libusb1-backed transport for one opened DSLogic."""

    def __init__(self, context, device, handle):
        self._ctx = context
        self._device = device
        self._handle = handle
        self._claimed = False

    # --- discovery -------------------------------------------------------
    @staticmethod
    def list_devices():
        """Return [(usb1 context, device, superspeed)] for every DreamSourceLab device."""
        import usb1
        ctx = usb1.USBContext()
        ctx.open()
        found = []
        for dev in ctx.getDeviceIterator(skip_on_error=True):
            if dev.getVendorID() != C.DS_VENDOR_ID:
                continue
            speed = dev.getDeviceSpeed()
            if speed not in (usb1.SPEED_HIGH, usb1.SPEED_SUPER, getattr(usb1, "SPEED_SUPER_PLUS", -1)):
                continue
            found.append((ctx, dev, speed != usb1.SPEED_HIGH))
        return found

    @classmethod
    def open_device(cls, ctx, dev) -> "UsbTransport":
        import usb1
        try:
            handle = dev.open()
        except usb1.USBErrorNotSupported as e:
            raise DSLogicError(
                "libusb cannot open the analyser: no WinUSB/libusb driver bound. "
                "Install DSView once (its installer adds the driver) or use Zadig to bind WinUSB.") from e
        except usb1.USBErrorAccess as e:
            raise DSLogicError(
                "Access denied opening the analyser. Close DSView (it holds the device); "
                "on Linux add the DreamSourceLab udev rule (see README).") from e
        t = cls(ctx, dev, handle)
        return t

    def claim(self) -> None:
        import usb1
        try:
            if hasattr(self._handle, "setAutoDetachKernelDriver"):
                try:
                    self._handle.setAutoDetachKernelDriver(True)
                except usb1.USBError:
                    pass
            self._handle.claimInterface(C.USB_INTERFACE)
        except usb1.USBErrorBusy as e:
            raise DSLogicError("USB interface busy: another program (DSView?) is using the analyser.") from e
        self._claimed = True

    @property
    def bus_address(self) -> str:
        return f"{self._device.getBusNumber()}.{self._device.getDeviceAddress()}"

    def strings(self) -> tuple[str, str]:
        d = self._device
        return (self._handle.getASCIIStringDescriptor(d.getManufacturerDescriptor()) or "",
                self._handle.getASCIIStringDescriptor(d.getProductDescriptor()) or "")

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            if self._claimed:
                self._handle.releaseInterface(C.USB_INTERFACE)
        except Exception:
            pass
        self._handle.close()
        self._handle = None
        self._claimed = False

    # --- control commands (command.c) -------------------------------------
    def ctl_write(self, dest: int, offset: int = 0, data: bytes = b"") -> None:
        import usb1
        payload = struct.pack("<BHB", dest, offset, len(data)) + bytes(data)
        try:
            self._handle.controlWrite(usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE, C.CMD_CTL_WR,
                                      0, 0, payload, timeout=C.CTL_TIMEOUT_MS)
        except usb1.USBError as e:
            raise DSLogicError(f"CMD_CTL_WR dest={dest} offset={offset:#x} failed: {e}") from e

    def ctl_read(self, dest: int, offset: int = 0, size: int = 1) -> bytes:
        import usb1
        header = struct.pack("<BHB", dest, offset, size)
        try:
            self._handle.controlWrite(usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE, C.CMD_CTL_RD_PRE,
                                      0, 0, header, timeout=C.CTL_TIMEOUT_MS)
            time.sleep(0.010)     # command.c waits 10 ms between the two halves
            data = self._handle.controlRead(usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE, C.CMD_CTL_RD,
                                            0, 0, size, timeout=C.CTL_TIMEOUT_MS)
        except usb1.USBError as e:
            raise DSLogicError(f"CMD_CTL_RD dest={dest} offset={offset:#x} failed: {e}") from e
        return bytes(data)

    def bulk_write(self, data: bytes, timeout_ms: int = 1000) -> int:
        import usb1
        try:
            return self._handle.bulkWrite(C.EP_BULK_OUT, data, timeout=timeout_ms)
        except usb1.USBErrorTimeout as e:
            return e.transferred if hasattr(e, "transferred") else 0
        except usb1.USBError as e:
            raise DSLogicError(f"bulk OUT failed: {e}") from e

    # --- async bulk IN ---------------------------------------------------
    def submit_read(self, size: int, callback: Callable, timeout_ms: int = 0) -> ReadRequest:
        """Queue a bulk-IN read of `size` bytes. `callback(data, status)` returns True to
        resubmit the same transfer (DSView's resubmit_transfer), False to retire it."""
        import usb1
        transfer = self._handle.getTransfer()
        req = _Usb1Read(transfer)
        status_names = {usb1.TRANSFER_COMPLETED: "ok", usb1.TRANSFER_TIMED_OUT: "timeout",
                        usb1.TRANSFER_CANCELLED: "cancelled"}

        def on_done(tr):
            n = tr.getActualLength()
            data = memoryview(tr.getBuffer())[:n]
            again = callback(data, status_names.get(tr.getStatus(), "error"))
            if again:
                try:
                    tr.submit()
                    return
                except usb1.USBError:
                    pass
            req.done = True

        transfer.setBulk(C.EP_BULK_IN, size, callback=on_done, timeout=timeout_ms)
        transfer.submit()
        return req

    def handle_events(self, timeout_s: float) -> None:
        self._ctx.handleEventsTimeout(timeout_s)


class _Usb1Read(ReadRequest):
    def __init__(self, transfer):
        self.transfer = transfer
        self.done = False

    def cancel(self) -> None:
        if not self.done:
            try:
                self.transfer.cancel()
            except Exception:
                pass
