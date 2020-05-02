from flask_restplus import Api
from flask import Blueprint
from .main.controller.client_address_search_controller import api as ns_address_search
from .main.controller.client_profile_summarization_controller import api as ns_profile_summarization
blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='AML Utility API',
          version='1.0',
          description='flask restplus web service for AML'
          )

api.add_namespace(ns_address_search, path='/client')
api.add_namespace(ns_profile_summarization, path='/client')
