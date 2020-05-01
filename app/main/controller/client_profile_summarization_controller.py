from flask_restplus import Resource
from ..service.client_profile_summarization_service import bert_summarizer
from ..util.client_profile_summarization import ClientProfileSummarization

api = ClientProfileSummarization.api
payload = ClientProfileSummarization.payload


@api.route("/get-summary")
@api.response(200, "Success")
@api.response(400, "Bad Response")
@api.response(404, "Not Found")
@api.response(500, "Internal Server error")
class ProfileSummarization(Resource):
    '''
    Performs Address Search on Web Data for Client Profile
    '''
    @api.doc("Get Client Profile Summary")
    @api.expect(payload, validate=True)
    def post(self):
        # noinspection PyBroadException
        try:
            client_name = self.api.payload['ProfileSummaryBenchmark']
            search_url_list = self.api.payload['SearchUrlList']
            bert_summarizer(source_url_list=search_url_list, kyc_doc=client_name)
        except Exception as e:
            self.api.abort(500)
