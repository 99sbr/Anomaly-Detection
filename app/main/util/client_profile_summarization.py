from flask_restplus import Namespace, fields


class ClientProfileSummarization:
    api = Namespace('ClientProfileSummarization', description='Client Profile Summarization')
    payload = api.model('ClientProfileSummarization',
                        {
                            'KYC-Documentum': fields.String(required=True,
                                                            description='from documentum get profile summary'),
                            'SearchUrlList': fields.List(fields.String, required=True,
                                                         description='List of URL to search')
                        })
