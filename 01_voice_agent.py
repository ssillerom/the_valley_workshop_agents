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

logger = logging.getLogger("basic-agent")

load_dotenv(dotenv_path=".env")


class AgenteValley(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Tu nombre es Carolina. Interactúas con los usuarios a través de la voz, "
            "por lo tanto mantén tus respuestas concisas y directas. "
            "Eres curiosa y amigable, y tienes sentido del humor. "
            "Hablas en español en todo momento.",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply(instructions="greet the user and ask about their day")

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
        latitude: str,
        longitude: str,
    ):
        """Called when the user asks for weather related information.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location
            longitude: The longitude of the location
        """

        logger.info(f"Looking up weather for {location}")

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
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.4),
        stt=deepgram.STT(model="nova-3-general", language="es"),
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
        # use LiveKit's turn detection model
        turn_detection=turn_detector.EOUModel(),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

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