import json
import boto3

# Bedrock Runtime client used to invoke and question the models
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  # Replace with your desired region
)

def lambda_handler(event, context):
    lite_prompt = "2 difference between AWS DynamoDB and AWS Redis"

    body = json.dumps({
        "inputText": lite_prompt,
        "textGenerationConfig": {
            "maxTokenCount": 128,
            "stopSequences": [],
            "temperature": 0,
            "topP": 0.9
        }
    })

    try:
        # The actual call to retrieve a response from the model
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="amazon.titan-text-lite-v1",  # Replace with your model ID
            accept='application/json',
            contentType='application/json'
        )

        response_body = json.loads(response['Payload'].read().decode('utf-8'))

        # Check if the response has the expected structure
        if 'results' in response_body and response_body['results']:
            outputText = response_body['results'][0].get('outputText', '')
            
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Headers': '*',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
                },
                'body': outputText
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": "Invalid Bedrock model response format"})
            }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }