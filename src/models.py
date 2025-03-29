from dataclasses import dataclass, field
from livekit.agents.voice import Agent
from typing import Optional
import yaml


@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None

    reservation_time: Optional[str] = None

    order: Optional[list[str]] = None

    customer_credit_card: Optional[str] = None
    customer_credit_card_expiry: Optional[str] = None
    customer_credit_card_cvv: Optional[str] = None

    expense: Optional[float] = None
    checked_out: Optional[bool] = None

    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        data = {
            "nombre_cliente": self.customer_name or "desconocido",
            "telefono_cliente": self.customer_phone or "desconocido",
            "hora_reserva": self.reservation_time or "desconocido",
            "pedido": self.order or "desconocido",
            "tarjeta_credito": {
            "numero": self.customer_credit_card or "desconocido",
            "caducidad": self.customer_credit_card_expiry or "desconocido",
            "cvv": self.customer_credit_card_cvv or "desconocido",
            }
            if self.customer_credit_card
            else None,
            "importe": self.expense or "desconocido",
            "pagado": self.checked_out or False,
        }
        # summarize in yaml performs better than json
        return yaml.dump(data)
