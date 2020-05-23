import unittest

from flask import current_app
from flask_testing import TestCase

from manage import app


class TestTestingConfig(TestCase):
    def create_app(self):
        app.config.from_object('app.main.config.TestingConfig')
        return app

    def test_app_is_testing(self):
        self.assertTrue(app.config['DEBUG'] is True)
        self.assertFalse(current_app is None)


if __name__ == '__main__':
    unittest.main()
