"""
Gestão de energia do servidor local.

Num portátil com Windows, a suspensão automática pode pausar o processamento a
meio de uma resposta. Enquanto a API corre, pedimos ao sistema que não suspenda
por inactividade e avisamos quando o computador está a funcionar a bateria, o
que reduz a velocidade do CPU. Nos outros sistemas estas funções não fazem nada.
"""

import ctypes
import logging
import sys

logger = logging.getLogger(__name__)

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def prevent_system_sleep() -> None:
    """Impede a suspensão automática enquanto a thread actual estiver activa."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)


def allow_system_sleep() -> None:
    """Devolve ao sistema o controlo normal da suspensão."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


def warn_if_on_battery() -> None:
    """Regista um aviso se o computador não estiver ligado à corrente."""
    if sys.platform != "win32":
        return
    status = _SystemPowerStatus()
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)) and status.ACLineStatus == 0:
        logger.warning(
            "O computador está a funcionar a bateria: as respostas serão mais lentas. "
            "Ligue-o à corrente para obter o melhor desempenho."
        )
