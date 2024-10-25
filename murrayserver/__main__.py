
from .server import Server

from asyncio import get_event_loop


async def start():
    server = Server()
    await server.start()


if __name__ == '__main__':

    print('Laiton: Checking Main Works')
    try:
        loop = get_event_loop()
        loop.run_until_complete(start())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
# from .server import Server
# import asyncio  # Import asyncio to use asyncio.run

# async def start():
#     server = Server()
#     await server.start()

# if __name__ == '__main__':
#     print('Laiton: Checking Main Works')
#     try:
#         # Use asyncio.run to manage the event loop
#         asyncio.run(start())
#     except KeyboardInterrupt:
#         pass
