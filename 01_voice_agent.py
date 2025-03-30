import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero, turn_detector, elevenlabs
from livekit.plugins import noise_cancellation
import os

logger = logging.getLogger("el nombre de tu agente")

load_dotenv(dotenv_path=".env")


class AgenteValley(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Añade el prompt que quiera que el agente siga",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply(instructions="greet the user and ask about their day")

    # todas las funciones anotadas con @function_tool se pasarán al LLM cuando este
    # agente esté activo
    @function_tool
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
        latitude: str,
        longitude: str,
    ):
        """Se llama cuando el usuario solicita información relacionada con el clima.
        Cuando se proporciona una ubicación, por favor estima la latitud y longitud de la ubicación y
        no le pidas al usuario que las proporcione.

        Args:
            location: La ubicación sobre la que están preguntando
            latitude: La latitud de la ubicación
            longitude: La longitud de la ubicación
        """

        logger.info(f"Buscando el tiempo en: {location}")

        return {
            "weather": "sunny",
            "temperature": 70,
            "location": location,
        }


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # usa cualquier combinacion de STT, LLM, TTS
        llm=...,
        stt=...,
        tts=...,
        # use LiveKit's turn detection model
        turn_detection=turn_detector.EOUModel(),
    )

    # las log metrics se emiten una vez ha terminado la sesión
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Uso: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    # wait for a participant to join the room
    await ctx.wait_for_participant()

    await session.start(
        agent=AgenteValley(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
        noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))