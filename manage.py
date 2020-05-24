import unittest

from flask_script import Manager
from waitress import serve

from app import blueprint
from app.main import create_app

app = create_app('test')
app.register_blueprint(blueprint)
manager = Manager(app)


@manager.command
def run():
    serve(app, port=1234)


@manager.command
def test():
    """
    Runs the unit tests.
    """
    tests = unittest.TestLoader().discover('app/test', pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if result.wasSuccessful():
        return 0
    return 1


if __name__ == '__main__':
    manager.run()
