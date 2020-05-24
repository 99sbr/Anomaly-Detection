from flask_restplus import Namespace, fields


class ClientAddressSearch:
    api = Namespace('ClientAddressSearch', description='Client Address Search')
    payload = api.model('ClientAddressSearch',
                        {
                            'ClientName': fields.String(required=True, description='client name'),
                            'SearchUrlList': fields.List(fields.String, required=True,
                                                         description='List of URL to search')
                        })
