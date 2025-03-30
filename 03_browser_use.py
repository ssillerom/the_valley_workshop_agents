from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
import asyncio
# Configura el path de tu navegador
browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        # chrome_instance_path='/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',  # macOS path
        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        # For Linux, typically: '/usr/bin/google-chrome'
    )
)

# Crear el agente con el navegador configurado
agent = Agent(
    task="LA TAREA QUE QUIERAS QUE HAGA BROWSER USE POR TI",
    llm=ChatOpenAI(model='gpt-4o'),
    browser=browser,
)

async def main():
    await agent.run()

    input('Pulsa enter para cerrar el navegador...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())