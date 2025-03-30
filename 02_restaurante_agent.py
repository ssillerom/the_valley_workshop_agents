import logging

from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field
from src.models import UserData

from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import deepgram, openai, silero, elevenlabs, turn_detector, noise_cancellation
import os

logger = logging.getLogger("restaurant-el nombre de tu RESTAURANTE")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=".env")

RunContext_T = RunContext[UserData]


# Funciones comunes para los agentes


@function_tool()
async def update_name(
    name: Annotated[str, Field(description="El nombre del cliente")],
    context: RunContext_T,
) -> str:
    """Se llama cuando el usuario proporciona su nombre.
    Confirma la ortografía con el usuario antes de llamar a la función."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"El nombre ha sido actualizado a {name}"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="El número de teléfono del cliente")],
    context: RunContext_T,
) -> str:
    """Se llama cuando el usuario proporciona su número de teléfono.
    Confirma los dígitos con el usuario antes de llamar a la función."""
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"El número de teléfono ha sido actualizado a {phone}"


@function_tool()
async def to_greeter(context: RunContext_T) -> Agent:
    """Se llama cuando el usuario hace preguntas no relacionadas o solicita
    otros servicios que no están en su descripción de trabajo."""
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entrando en la tarea del agente: {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # add the previous agent's chat history to the current agent
        if userdata.prev_agent:
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # add an instructions including the user data as a system message
        chat_ctx.add_message(
            role="system",
            content=f"Eres el agente {agent_name}. Los datos actuales del usuario son {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transfiriendo a {name}."

    def _truncate_chat_ctx(
        self,
        items: list[llm.ChatItem],
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list[llm.ChatItem]:
        """Trunca el contexto de chat para mantener solo los últimos n mensajes.
        
        Args:
            items: Lista de elementos del chat a truncar.
            keep_last_n_messages: Número de mensajes más recientes a conservar.
            keep_system_message: Si se deben mantener los mensajes del sistema.
            keep_function_call: Si se deben mantener las llamadas a funciones.
            
        Returns:
            Lista truncada de elementos de chat.
        """

        def _valid_item(item: llm.ChatItem) -> bool:
            if not keep_system_message and item.type == "message" and item.role == "system":
                return False
            if not keep_function_call and item.type in [
                "function_call",
                "function_call_output",
            ]:
                return False
            return True

        new_items: list[llm.ChatItem] = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items = new_items[::-1]

        # the truncated items should not start with function_call or function_call_output
        while new_items and new_items[0].type in ["function_call", "function_call_output"]:
            new_items.pop(0)

        return new_items


class Greeter(BaseAgent):
    """
    Agente de bienvenida para el restaurante.

    Este agente se encarga de dar la bienvenida a los usuarios y redirigirlos
    a otros agentes según sus necesidades, como hacer una reserva o realizar
    un pedido para llevar.

    Attributes:
        menu (str): El menú del restaurante que puede compartir con los usuarios.

    Methods:
        to_reservation: Transfiere la conversación al agente de reservas cuando
                       el usuario desea hacer una reserva.
        to_takeaway: Transfiere la conversación al agente de pedidos para llevar
                    cuando el usuario desea realizar un pedido.
    """
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=("añade prompts de conversación para el agente de bienvenida "
            ),
            llm=openai.LLM(model="gpt-4o-mini", parallel_tool_calls=False),
            tts=elevenlabs.TTS(voice=elevenlabs.tts.Voice(...),
        )
        self.menu = menu

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> Agent:
        """Se llama cuando el usuario quiere hacer una reserva.
        Esta función gestiona la transición al agente de reservas
        que recopilará los detalles necesarios como la hora de la reserva,
        el nombre del cliente y el número de teléfono."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> Agent:
        """Se llama cuando el usuario quiere hacer un pedido para llevar.
        Esto incluye gestionar pedidos para recoger, entrega a domicilio, o cuando el usuario
        quiere proceder al pago con su pedido existente."""
        return await self._transfer_to_agent("takeaway", context)


class Reservation(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="añade prompt de conversación para el agente de reservas",
            tools=[update_name, update_phone, to_greeter],
            tts=elevenlabs.TTS(...),
        )

    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="La hora de la reserva")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.reservation_time = time
        return f"La hora de la reserva se ha actualizado a {time}"

    @function_tool()
    async def confirm_reservation(self, context: RunContext_T) -> str | tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Por favor, proporcione su nombre y número de teléfono primero."

        if not userdata.reservation_time:
            return "Por favor, proporcione la hora de la reserva primero."

        return await self._transfer_to_agent("greeter", context)


class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=("añade prompt de conversación para el agente de comida para llevar "
            ),
            tools=[to_greeter],
            tts=elevenlabs.TTS(...),
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="Los elementos del pedido completo")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.order = items
        return f"El pedido se ha actualizado a {items}"

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.order:
            return "No se ha encontrado ningún pedido para llevar. Por favor, haga un pedido primero."

        return await self._transfer_to_agent("checkout", context)


class Checkout(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                "añade prompt de conversación para el agente de checkout "
                "incluyendo la confirmación del importe, la tarjeta de crédito y el pago"
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=elevenlabs.TTS(voice=elevenlabs.tts.Voice(...),
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="El coste del pedido")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.expense = expense
        return f"El coste del pedido es: {expense}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="La tarjeta de crédito")],
        expiry: Annotated[str, Field(description="Fecha de caducidad de la tarjeta de Crédito")],
        cvv: Annotated[str, Field(description="CVV de la tarjeta de crédito")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"El número de la tarjeta de crédito ha sido actualizado: {number}"

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.expense:
            return "Por favor confirme el importe primero."

        if (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Por favor proporcione la información de la tarjeta de crédito primero."

        userdata.checked_out = True
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        return await self._transfer_to_agent("takeaway", context)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    menu = "INTRODUCE AQUI TU MENÚ DE RESTAURANTE SEPARADO POR COMAS Y CON PRECIO"  # load the menu from a file or database
    # load the menu from a file or database
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(menu),
            "reservation": Reservation(),
            "takeaway": Takeaway(menu),
            "checkout": Checkout(menu),
        }
    )
    agent = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(#???),
        llm=openai.LLM(#???),
        tts=elevenlabs.TTS(#???),
        vad=silero.VAD.load(),
        max_tool_steps=5,
    )

    await agent.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
        room_input_options=RoomInputOptions(
        # noise_cancellation=noise_cancellation.BVC(), solo usar si tienes linux o macOs
        ),
    )

    await agent.say("Restaurante TU NOMBRE DE RESTAURANTE, ¿Dígame?") #Mensaje de inicio


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))