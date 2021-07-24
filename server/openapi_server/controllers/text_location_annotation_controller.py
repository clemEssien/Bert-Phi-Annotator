import connexion
from openapi_server.models.error import Error  # noqa: E501
from openapi_server.models.text_location_annotation import TextLocationAnnotation  # noqa: E501
from openapi_server.models.text_location_annotation_request import TextLocationAnnotationRequest  # noqa: E501
from openapi_server.models.text_location_annotation_response import TextLocationAnnotationResponse  # noqa: E501
from openapi_server import nlp_config as cf


def create_text_location_annotations():  # noqa: E501
    """Annotate locations in a clinical note

    Return the location annotations found in a clinical note # noqa: E501

    :param text_location_annotation_request:
    :type text_location_annotation_request: dict | bytes

    :rtype: TextLocationAnnotationResponse
    """
    res = None
    status = None
    if connexion.request.is_json:
        try:
            annotation_request = TextLocationAnnotationRequest.from_dict(connexion.request.get_json())  # noqa: E501
            note = annotation_request._note
            annotations = []
            result = cf.get_entities("./dslim-bert/tokenizer", "./dslim-bert/model", note.text)
            
            for output in result:
                if 'LOC' in output['entity']:
                    annotations.append(TextLocationAnnotation(
                            start=int(output['start']),
                            length=len(output['word']),
                            text=output['word'],
                            confidence=round(float(output['score']*100), 2)
                        ))
            res = TextLocationAnnotationResponse(annotations)
            status = 200
        except Exception as error:
            status = 500
            print(error)
            res = Error("Internal error", status, str(error))
    return res, status
