import json
import boto3

# Bedrock Runtime client used to invoke and question the models
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  # Replace with your desired region
)

def lambda_handler(event, context):
    
    # PROVIDE your prompt here
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
    # The actual call to retrieve a response from the model
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-text-lite-v1",  # Replace with your model ID
        accept='application/json',
        contentType='application/json'
        )
    
    response_body = json.loads(response.get('body').read())
    response_text = response_body.get('results')[0].get('outputText')
    parse_text = response_text[response_text.index('\n')+1:]
    model_completion = parse_text.strip()
    
    #  This code will send a respone of Text Completion with stats
    return {
        'statusCode': 200,
        'body': json.dumps(model_completion)
    }