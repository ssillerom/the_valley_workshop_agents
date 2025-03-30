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

logger = logging.getLogger("restaurant-magalia")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=".env")

RunContext_T = RunContext[UserData]


# common functions


@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"The name is updated to {name}"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"The phone number is updated to {phone}"


@function_tool()
async def to_greeter(context: RunContext_T) -> Agent:
    """Called when user asks any unrelated questions or requests
    any other services not in your job description."""
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

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

        return next_agent, f"Transferring to {name}."

    def _truncate_chat_ctx(
        self,
        items: list[llm.ChatItem],
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list[llm.ChatItem]:
        """Truncate the chat context to keep the last n messages."""

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
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
            f"Eres un amable recepcionista de restaurante. El menú es: {menu}\n"
            "Tu trabajo es saludar a quien llama y entender si quieren "
            "hacer una reserva o pedir comida para llevar. Guíalos al agente adecuado usando las herramientas."
            ),
            llm=openai.LLM(model="gpt-4o-mini", parallel_tool_calls=False),
            tts=elevenlabs.TTS(voice=elevenlabs.tts.Voice(
            id=os.getenv("ELEVEN_VOICE_ID"),
            name="Carolina",
            category="premade",
            settings=elevenlabs.tts.VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
        )),
        )
        self.menu = menu

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> Agent:
        """Called when user wants to make a reservation.
        This function handles transitioning to the reservation agent
        who will collect the necessary details like reservation time,
        customer name and phone number."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> Agent:
        """Called when the user wants to place a takeaway order.
        This includes handling orders for pickup, delivery, or when the user wants to
        proceed to checkout with their existing order."""
        return await self._transfer_to_agent("takeaway", context)


class Reservation(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Eres un agente de reservas en un restaurante. Tu trabajo es preguntar por "
            "la hora de la reserva, luego el nombre del cliente, y el número de teléfono. Después "
            "confirma los detalles de la reserva con el cliente.",
            tools=[update_name, update_phone, to_greeter],
            tts=elevenlabs.TTS(voice=elevenlabs.tts.Voice(
            id=os.getenv("ELEVEN_VOICE_ID"),
            name="Carolina",
            category="premade",
            settings=elevenlabs.tts.VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
        )),
        )

    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.reservation_time = time
        return f"The reservation time is updated to {time}"

    @function_tool()
    async def confirm_reservation(self, context: RunContext_T) -> str | tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number first."

        if not userdata.reservation_time:
            return "Please provide reservation time first."

        return await self._transfer_to_agent("greeter", context)


class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
            f"Eres un agente de comida para llevar que toma pedidos de los clientes. "
            f"Nuestro menú es: {menu}\n"
            "Aclara peticiones especiales y confirma el pedido con el cliente."
            ),
            tools=[to_greeter],
            tts=elevenlabs.TTS(voice=elevenlabs.tts.Voice(
            id=os.getenv("ELEVEN_VOICE_ID"),
            name="Carolina",
            category="premade",
            settings=elevenlabs.tts.VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
        )),
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.order = items
        return f"The order is updated to {items}"

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.order:
            return "No takeaway order found. Please make an order first."

        return await self._transfer_to_agent("checkout", context)


class Checkout(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
            f"Eres un agente para realizar pagos en un restaurante. El menú es: {menu}\n"
            "Tu responsabilidad es confirmar el coste total del "
            "pedido y luego recopilar el nombre del cliente, número de teléfono e información "
            "de la tarjeta de crédito, incluyendo el número de tarjeta, fecha de caducidad y CVV paso a paso."
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=elevenlabs.TTS(voice=elevenlabs.tts.Voice(
            id=os.getenv("ELEVEN_VOICE_ID"),
            name="Carolina",
            category="premade",
            settings=elevenlabs.tts.VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
            )),
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.expense = expense
        return f"The expense is confirmed to be {expense}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"The credit card number is updated to {number}"

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."

        if (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Please provide the credit card information first."

        userdata.checked_out = True
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        return await self._transfer_to_agent("takeaway", context)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
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
        stt=deepgram.STT(model="nova-3-general", language="es"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(voice=elevenlabs.tts.Voice(
                id=os.getenv("ELEVEN_VOICE_ID"),
                name="Carolina",
                category="premade",
                settings=elevenlabs.tts.VoiceSettings(
                    stability=0.71,
                    similarity_boost=0.5,
                    style=0.0,
                    use_speaker_boost=True
                )
        )),
        vad=silero.VAD.load(),
        max_tool_steps=5,
    )

    await agent.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
        room_input_options=RoomInputOptions(
        noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await agent.say("Restaurante Magalia, ¿Dígame?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))