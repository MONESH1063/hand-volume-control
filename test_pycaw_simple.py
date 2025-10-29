# test_pycaw_simple.py
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import comtypes

comtypes.CoInitialize()  # ensure COM initialized

try:
    # Preferred modern method (pycaw stable)
    device = AudioUtilities.GetSpeakers()
    try:
        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    except AttributeError:
        # fallback if Activate not present
        interface = device._ctl.QueryInterface(IAudioEndpointVolume)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    rng = volume.GetVolumeRange()
    print("OK: got volume interface. Range:", rng)
    cur = volume.GetMasterVolumeLevel()
    print("Current dB:", cur)
    # set to midpoint as a quick test:
    target = rng[0] + (rng[1] - rng[0]) * 0.5
    volume.SetMasterVolumeLevel(target, None)
    print("Set volume to 50% (dB value).")
except Exception as e:
    print("ERROR:", repr(e))
