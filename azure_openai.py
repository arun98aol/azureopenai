from retry import retry
from tenacity import retry, stop_after_attempt, wait_fixed, RetryCallState
from openai import AzureOpenAI
import openai  
import json, os, sys
from openai.types.chat.completion_create_params import ResponseFormat
from apps.cx_network.functions.beta_testing import update_token_usage
from flask import session

import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_json(client, json_string):
    try:
        # Attempt to load the JSON
        json_object = json.loads(json_string)
        # Dump it back to string with proper formatting
        fixed_json_string = json.dumps(json_object, indent=2)
        return fixed_json_string
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")

        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="cx-gpt-4",
                messages=[
                    {"role": "system", "content": "You are a JSON schema validator and expert."},
                    {"role": "user", "content": "Please fix the error in my JSON string" + json_string},
                ],
                temperature=0,
                response_format = ResponseFormat(type="json_object")
            )

        return response.choices[0].message.content
  
def append_to_jsonl(file_path, data):  
    with open(file_path, "a", encoding="utf-8") as f:  
        f.write(json.dumps(data) + "\n")  
    
def update_usage(usage, fn_name, doc_id=None, doc_type=None):
    try:
        status = update_token_usage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            function_name=fn_name, 
            user_local_timestamp=session["user_local_time"], 
            user_id=session["user_id"],
            doc_id=doc_id,
            doc_type=doc_type
        )
        return status
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        logger.error(f"{exc_type}, {fname}, {exc_tb.tb_lineno}")
        error_message = f'Error: {str(e)}'
        return False 
  
# Define your client configurations  
clients_config = {
        "cx-gpt-4o": [
            {  # EAST3
            "api_key": os.getenv("AZURE_OPENAI_API_KEY4"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT4")
            },
            {  # EASTUS2
            "api_key": os.getenv("AZURE_OPENAI_API_KEY2"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT2")
            },
            {  # WESTUS
            "api_key": os.getenv("AZURE_OPENAI_API_KEY3"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT3")
            }
        ], 
        "cx-gpt-4o-mini": [
            { # EASTUS 
                "api_key": os.getenv("AZURE_OPENAI_API_KEY4"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT4")
            }
        ],
        "cx-gpt-4o-08-2024": [
            { # EASTUS 
                "api_key": os.getenv("AZURE_OPENAI_API_KEY4"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT4")
            }
        ]
    }

def refine_prompt(bw_message: list) -> str:
    """
    Refines the content of the last user message using a smaller model to avoid triggering content filters.
    """
    try:
        user_message = bw_message[-1]["content"]
        logger.info(f"Refining user message: {user_message}")

        # Define client_config for the refinement model
        client_config = clients_config["cx-gpt-4o-mini"][0]

        client = AzureOpenAI(
            api_key=client_config["api_key"],
            api_version="2024-10-21",
            azure_endpoint=client_config["endpoint"]
        )
        response = client.chat.completions.create(
            model="cx-gpt-4o-mini",
            seed=274,
            messages=[{"role": "user", "content": f"Please rephrase the following message while keeping its meaning the same. Don't change anything in triple quotes. Output the rephrased message.\n\n{user_message}"}],
            temperature=0,
            max_tokens=250
        )

        refined_message = response.choices[0].message.content
        logger.info(f"Refined message: {refined_message}")
        return refined_message
    except Exception as e:
        logger.error(f"Error during prompt refinement: {e}")
        return user_message  # Fallback to original message if refinement fails

def handle_retry(retry_state: RetryCallState):
    """
    Handles the retry logic before sleeping, refining the user message if necessary.
    """
    exception = retry_state.outcome.exception()
    exception = exception.__dict__ if exception else None
    logger.info(f"Retry attempt {retry_state.attempt_number}: {exception}")
    # You can log the type of the exception to confirm it's an APIError
    # logger.debug(f"Exception type: {type(exception)}")
    # Extract bw_message from the arguments (assuming it's the third positional argument)
    # Check for bw_message in both positional and keyword arguments
    bw_message = None
    if retry_state.args is not None:
        if len(retry_state.args) > 2:
            bw_message = retry_state.args[2]
    if bw_message is None and retry_state.kwargs is not None:
        if 'bw_message' in retry_state.kwargs:
            bw_message = retry_state.kwargs['bw_message']
    else:
        return

    logger.info(f"bw_message: {bw_message}")

    if bw_message is None:
        logger.warning("bw_message is None. Cannot refine the user message.")
        return

    # if isinstance(exception, openai.APIError) and bw_message:
    if exception and bw_message:
        logger.info(f"OpenAI APIError: True")
        # Access the error response from exception.body instead of exception.args[0]
        error_response = exception.get("body")
        logger.info(f"APIError body: {error_response}")
        if (
            isinstance(error_response, dict)
            and error_response.get("error", {}).get("innererror", {}).get("code") == "ResponsibleAIPolicyViolation"
        ):
            logger.warning("Detected ResponsibleAIPolicyViolation. Refining the user message.")
            refined_message = refine_prompt(bw_message)
            bw_message[-1]["content"] = refined_message

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    before_sleep=handle_retry
)
def call_openai_api(
        model: str,
        seed: int,
        bw_message: list,
        tools: list = None,
        tool_choice: dict = None,
        temperature: float = 0,
        max_tokens: int = 500,
        train_save: bool = False,
        doc_name: str = None,
        doc_id: str = None,
        fn_name: str = None,
        parallel_tool_calls: bool = False
    ):

    # Keep track of attempts using an attribute of the function
    if not hasattr(call_openai_api, 'attempt_counter'):
        call_openai_api.attempt_counter = 0

    for client_config in clients_config[model]:
        try:
            # Manually trigger the APIError on the first attempt
            # if call_openai_api.attempt_counter == 0:
            #     call_openai_api.attempt_counter += 1
            #     error_response = {
            #         "error": {
            #             "message": "The response was filtered due to the prompt triggering Azure OpenAI's content management policy.",
            #             "code": "content_filter",
            #             "innererror": {
            #                 "code": "ResponsibleAIPolicyViolation",
            #                 "content_filter_result": {
            #                     "jailbreak": {"filtered": True, "detected": True}
            #                 }
            #             }
            #         }
            #     }
            #     raise openai.APIError(
            #         message=error_response["error"]["message"],
            #         request=None,
            #         body=error_response
            #     )

            client = AzureOpenAI(
                api_key=client_config["api_key"],
                api_version="2024-10-21",
                azure_endpoint=client_config["endpoint"]
            )
            response = client.chat.completions.create(
                model=model,
                seed=seed,
                messages=bw_message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls
            )
            logger.info(f"Response: {response}")

            status = update_usage(
                usage=response.usage,
                fn_name=fn_name,
                doc_id=doc_id,
                doc_type=doc_name
            )

            if status is True:
                if tool_choice is None:
                    return response.choices[0].message.content
                elif len(response.choices[0].message.tool_calls) > 1 and parallel_tool_calls:
                    output_dict = {}
                    for tool_call in response.choices[0].message.tool_calls:
                        output_dict.update(json.loads(tool_call.function.arguments))
                else:
                    output = response.choices[0].message.tool_calls[0].function.arguments
                    output_dict = json.loads(output)

                if train_save:
                    complete_messages = bw_message + [{"role": "assistant", "content": output}]
                    formatted_data = {"document": f"{doc_name}_{doc_id}", "train_data": {"messages": complete_messages}}
                    json_file_path = f"data/training_data/{tool_choice['function']['name']}.jsonl"
                    append_to_jsonl(json_file_path, formatted_data)

                return output_dict
            else:
                logger.error("Token usage update failed.")
                return None

        except openai.APIError as e:
            logger.error(f"OpenAI API returned an API Error: {e}")
            raise  # Re-raise to allow the retry decorator to handle

        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}")
            raise  # Re-raise to allow the retry decorator to handle

        except openai.RateLimitError as e:
            logger.error(f"OpenAI API rate limit error: {e}")
            raise  # Re-raise to allow the retry decorator to handle

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise  # Re-raise to allow the retry decorator to handle

    raise Exception("All clients have failed to process the request.")

# @retry(tries=3, delay=2)  
def call_openai_api_east(  
        model: str,  
        seed: int,  
        bw_message: list,  
        tools: list = None,
        tool_choice: dict = None,  
        temperature: float = 0,  
        max_tokens: int = 500,
        train_save: bool = False,
        doc_name: str = None,
        doc_id: str = None
    ):  
    try:  
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY4"),  
            api_version="2024-07-01-preview",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT4")
        )
  
        response = client.chat.completions.create(  
            model=model,  
            seed=seed,  
            messages=bw_message,  
            temperature=temperature,  
            max_tokens=max_tokens,  
            top_p=1,  
            frequency_penalty=0,  
            presence_penalty=0,  
            tools=tools,  
            tool_choice=tool_choice,  
        )  

        if tool_choice is None:
            return response.choices[0].message.content
        else:
            output = response.choices[0].message.tool_calls[0].function.arguments  
        
        output_dict = json.loads(output)
        
        if train_save:
            # Construct the complete messages list including the assistant's response  
            complete_messages = bw_message + [{"role": "assistant", "content": output}]  

            # Format the data as required for training  
            formatted_data = {"document": f"{doc_name}_{doc_id}", "train_data": {"messages": complete_messages} } 

            # Append the formatted data to a jsonl file
            json_file_path = f"data/training_data/{tool_choice['function']['name']}.jsonl"
            append_to_jsonl(json_file_path, formatted_data)

        return output_dict        
          
    except openai.APIError as e:  
        print(f"\t- OpenAI API returned an API Error: {e}")  
        raise  
    except openai.APIConnectionError as e:  
        print(f"\t- OpenAI API connection error: {e}")  
        raise  
    except openai.RateLimitError as e:  
        print(f"\t- OpenAI API rate limit error: {e}")  
        raise  
    except Exception as e:  
        print(f"\t- An error occurred: {e}")  
        raise  
    
    
# @retry(tries=3, delay=2)  
def call_openai_api_1(  
        model: str,  
        seed: int,  
        bw_message: list,  
        tools: list,  
        tool_choice: dict,  
        temperature: float = 0,  
        max_tokens: int = 500
    ):  
    try:  
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY2"),  
            api_version="2024-07-01-preview",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT2")
        )
  
        response = client.chat.completions.create(  
            model=model,  
            seed=seed,  
            messages=bw_message,  
            temperature=temperature,  
            max_tokens=max_tokens,  
            top_p=1,  
            frequency_penalty=0,  
            presence_penalty=0,  
            tools=tools,  
            tool_choice=tool_choice,  
        )  
  
        output = response.choices[0].message.tool_calls[0].function.arguments  
  
        try:  
            return json.loads(output)  
        except json.JSONDecodeError:  
            print(f"\t- Error: Calling the function again to fix the JSON")  
            fixed_json = fix_json(client, output)  
            return json.loads(fixed_json)  
    except openai.APIError as e:  
        print(f"\t- OpenAI API returned an API Error: {e}")  
        raise  
    except openai.APIConnectionError as e:  
        print(f"\t- OpenAI API connection error: {e}")  
        raise  
    except openai.RateLimitError as e:  
        print(f"\t- OpenAI API rate limit error: {e}")  
        raise  
    except Exception as e:  
        print(f"\t- An error occurred: {e}")  
        raise  
    
import tiktoken

def get_token_count(input_text):
    token_count = len(tiktoken.tokenize(input_text))
    return token_count

# @retry(tries=3, delay=2)  
def call_openai_azureai(  
        model: str,
        bw_message: list,
        temperature: float = 0,
        max_tokens: int = 500,
        index_name: str = "tech-levels-index"   
    ):  
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY2"),  
        api_version="2024-07-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT2")
    )

    response = client.chat.completions.create(
        model=model,
        messages=bw_message,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={
            "data_sources":[
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
                        "index_name": index_name,
                        "authentication": {
                            "type": "api_key",
                            "key": os.getenv("AZURE_SEARCH_API_KEY"),
                        }
                    }
                }
            ],
        }
    )

    # output = response.choices[0].message.tool_calls[0].function.arguments  
    
    return response