from flask_restplus import Resource

from ..service.client_address_search_service import main_call
from ..util.client_address_search import ClientAddressSearch

api = ClientAddressSearch.api
payload = ClientAddressSearch.payload


@api.route("/get-address")
@api.response(200, "Success")
@api.response(400, "Bad Response")
@api.response(404, "Not Found")
@api.response(500, "Internal Server error")
class AddressSearch(Resource):
    """
    Performs Address Search on Web Data for Client Profile
    """



    @api.doc("Get Client Address")
    @api.expect(payload, validate=True)
    def post(self):

        # noinspection PyBroadException
        try:
            client_name = self.api.payload['ClientName']
            search_url_list = self.api.payload['SearchUrlList']
            address = main_call(base_url_list=search_url_list, client_name=client_name)
            return {'Response': address}
        except Exception as e:
            self.api.abort(500)
