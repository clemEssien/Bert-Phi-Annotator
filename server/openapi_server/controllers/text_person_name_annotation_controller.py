import connexion
from openapi_server.models.error import Error  # noqa: E501
from openapi_server.models.text_person_name_annotation_request import TextPersonNameAnnotationRequest  # noqa: E501
from openapi_server.models.text_person_name_annotation import TextPersonNameAnnotation  # noqa: E501
from openapi_server.models.text_person_name_annotation_response import TextPersonNameAnnotationResponse  # noqa: E501
from openapi_server.nlp_config import bert


def create_text_person_name_annotations():  # noqa: E501
    """Annotate person names in a clinical note

    Return the person name annotations found in a clinical note # noqa: E501

    :rtype: TextPersonNameAnnotationResponse
    """
    res = None
    status = None
    if connexion.request.is_json:
        try:
            annotation_request = TextPersonNameAnnotationRequest.from_dict(connexion.request.get_json())  # noqa: E501
            note = annotation_request._note  # noqa: E501
            annotations = []
            name_annotations = bert.get_entities(note.text, "PER")
            add_name_annotation(annotations, name_annotations)
            # for output in result:
            #     if 'PER' in output['entity']:
            #         annotations.append(TextPersonNameAnnotation(
            #                 start=int(output['start']),
            #                 length=len(output['word']),
            #                 text=output['word'],
            #                 confidence=round(float(output['score']*100), 2)
            #             ))
            res = TextPersonNameAnnotationResponse(annotations)
            status = 200
        except Exception as error:
            status = 500
            print(error)
            res = Error("Internal error", status, str(error))
    return res, status


def add_name_annotation(annotations, name_annotations):
    for match in name_annotations:
        annotations.append(
            TextPersonNameAnnotationResponse(
                           start=int(match['start']),
                           length=len(match['word']),
                           text=match['word'],
                           confidence=round(float(match['score']*100), 2)
        ))
