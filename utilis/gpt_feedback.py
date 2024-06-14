import glob
import base64
from mimetypes import guess_type
# from PIL import Image
from openai import AzureOpenAI
import ast


class ImageScorer:
    def __init__(self, api_base, api_key, deployment_name, api_version):
        self.api_base = api_base
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.client = self._create_client()

    def _create_client(self):
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            base_url=f"{self.api_base}/openai/deployments/{self.deployment_name}"
        )
        # return AzureOpenAI(
        #     openai_api_version=self.api_version,
        #     openai_api_key=self.api_key,
        #     openai_api_base=self.api_base,
        #     deployment_name=self.deployment_name,
        #     openai_api_type="azure",
        #     # base_url=f"{self.api_base}/openai/deployments/{self.deployment_name}",
        #     # temperature=0.7
        # )


    def local_image_to_data_url(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def get_score_for_images(self, map_index):
        file_pattern = f"./img/A2C/train/{map_index}-*.png"
        matched_files = glob.glob(file_pattern)
        good_url = self.local_image_to_data_url("./utilis/4398-33.png")

        score = 0
        for file_path in matched_files:
            try:
                data_url = self.local_image_to_data_url(file_path)
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        { "role": "system", "content": "You are a helpful assistant." },
                        { "role": "user", "content": [  
                            { 
                                "type": "text", 
                                "text": "Please provide a structured JSON response with a 'score' key. Please give a score (range -10 to 10) on the picture I send you in the following conversation with trajectories. A good trajectory in this environment is efficient and predictable, following a logical, consistent path while adapting smoothly to changes. It mimics natural human navigation with robustness, balancing movement evenly and optimizing energy use. The trajectory should be fluid and responsive, maintaining clarity and demonstrating purposeful intent. It balances symmetry and coordination, handling obstacles and rerouting seamlessly to ensure minimal resistance and a clear direction.Please mention that, the black pixels indicate the obstacles, while the white pixels indicate the the free space. Trajectories are blue, while the red lines indicate the agent met a collision and replanned the path. Please only return a total score based on those evaluation dimensions. The response should look like this: {'score': 3}" 
                            }
                        ] } ,
                        { "role": "user", "content": [
                            { 
                                "type": "text",
                                "text": "Here is an example of a good trajectory. It can receive a score of 9."
                            },
                            { 
                                "type": "image_url",
                                "image_url": {
                                    "url": good_url
                                }
                            }
                        ] },
                        { "role": "user", "content": [  
                            { 
                                "type": "text",
                                "text": "Here is the picture you need to rate."
                            },
                            { 
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ] } 
                    ],
                    max_tokens=300,
                    temperature=0,
                )
                print(response)
                content = response.choices[0].message.content
                # score_match = re.search("{'score': (\d+)}", content)
                # 解析为字典
                response_dict = ast.literal_eval(content)
                # 提取 score 值
                score = response_dict.get('score')
                # print(score_match)
                # if score_match:
                #     score = int(score_match.group(1))
                # else:
                #     print("Score not found in response.")

            except IOError:
                print(f"Unable to open image file: {file_path}")

        return score
    


# api_base = "https://ai-azureaiwestusengine881797519629.openai.azure.com/"
# api_key = '4d44030e3b2b4f978a44cccc49b464f1'
# deployment_name = 'gpt-4'
# api_version = '2023-03-15-preview'

# image_scorer = ImageScorer(api_base, api_key, deployment_name, api_version)
# map_index = 1950
# score = image_scorer.get_score_for_images(map_index)
