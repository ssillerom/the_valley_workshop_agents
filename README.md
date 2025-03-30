# Agentes de The Valley Workshop

<p align="center">
    <img src="https://thevalley.es/wp-content/uploads/2023/07/LOGO_TheValleyDBS_color-4.png" width="400" alt="The Valley Logo">
</p>

Este repositorio contiene ejemplos de agentes de IA que utilizan LiveKit y capacidades de automatización de navegador.

## Configuración

### Prerrequisitos

- Python 3.11 o superior
- Navegador (Chrome o Brave)
- Claves API para varios servicios

### Instalación
 
1. Clona el repositorio:
    ```bash
    git clone https://github.com/yourusername/the-valley-workshop-agents.git
    cd the-valley-workshop-agents
    ```

2. Configura un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3. Instala UV (si no lo tienes):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # O en Windows PowerShell
    # irm https://astral.sh/uv/install.ps1 | iex
    pip install uv
    ```

4. Instala las dependencias usando uv:
    ```bash
    uv sync
    ```

5. Configura las variables de entorno copiando el archivo de ejemplo y completando tus claves API:
    ```bash
    cp .env.example .env
    ```
    Luego edita el archivo `.env` con tus claves API para:
    - OpenAI
    - Deepgram
    - ElevenLabs
    - LiveKit

## Estructura del Proyecto

- **Agente de Voz** (`01_voice_agent.py`): Un agente conversacional que utiliza LiveKit para comunicación de audio en tiempo real.
- **Agente de Restaurante** (`02_restaurante_agent.py`): Un agente especializado para manejar reservas y pedidos de restaurante.
- **Automatización de Navegador** (`03_browser_use.py`): Un agente que puede controlar un navegador web para realizar tareas.

## Ejecutando los Ejemplos

### Agente de Voz

```bash
python 01_voice_agent.py
```

Este agente se conecta a una sala de LiveKit y participa en conversaciones de voz, proporcionando información meteorológica cuando se le solicita.

### Agente de Restaurante

```bash
python 02_restaurante_agent.py
```

Este agente simula un servicio de restaurante con múltiples agentes especializados:
- Recepcionista: Dirige a los clientes a reservas o comida para llevar
- Reservas: Gestiona la reserva de mesas
- Comida para llevar: Procesa pedidos de comida
- Pago: Gestiona el procesamiento de pagos

### Automatización de Navegador

```bash
python 03_browser_use.py
```

Este agente abre un navegador y realiza una tarea en LinkedIn. Puedes modificar la tarea en el script.

## Configuración

- Modifica las rutas del navegador en `03_browser_use.py` si es necesario
- Ajusta los comportamientos de los agentes editando sus instrucciones en los respectivos archivos
- Configura los ajustes de voz en la inicialización del agente para diferentes voces TTS

## Notas

- Todas las claves API deben mantenerse seguras y no comprometerse en el control de versiones
- Los agentes de voz requieren un micrófono y altavoces para su completa funcionalidad
- La automatización del navegador requiere un navegador compatible instalado en la ruta especificada

## Licencia

Este proyecto está licenciado bajo la Licencia Apache 2.0 - consulta el archivo LICENSE para más detalles.