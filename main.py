import os

from website import create_app


app = create_app()


if __name__ == '__main__':
    host = os.environ.get('APP_HOST', '127.0.0.1')
    port = int(os.environ.get('APP_PORT', '5000'))
    debug = os.environ.get('APP_DEBUG', '1').lower() in {'1', 'true', 'yes', 'on'}
    app.run(host=host, port=port, debug=debug)
