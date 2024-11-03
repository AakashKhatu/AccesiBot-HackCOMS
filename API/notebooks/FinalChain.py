# %%
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from typing import Any, Dict, Iterator, List, Optional
from together import Together
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, PrivateAttr
import json


class AccessibilityIssue(BaseModel):
    """Schema for accessibility issues."""

    Title: str = Field(description="Title of the issue")
    Description: str = Field(description="Detailed description of the issue")
    Suggestion: str = Field(description="Suggested fix for the issue")
    Importance_score: float = Field(description="Importance score between 0.0 and 1.0")


class AccessibilityResponse(BaseModel):
    """Schema for the complete accessibility response."""

    issues: Dict[str, AccessibilityIssue] = Field(
        description="Dictionary of accessibility issues"
    )


def load_context(file_path: str) -> str:
    """Load context from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


class TogetherVisionLLM(LLM):
    """A custom LLM that implements Together's Vision model capabilities.

    This LLM allows you to send both text and image inputs to Together's vision models,
    particularly designed for use with models like meta-llama/Llama-Vision-Free.

    Example:
        .. code-block:: python

            model = TogetherVisionLLM(
                model_name="meta-llama/Llama-Vision-Free",
                temperature=0.9,
                top_p=0.7
            )
            result = model.invoke(
                "Describe this image",
                image_url="https://example.com/image.png"
            )
    """

    model_name: str = Field(
        default="meta-llama/Llama-Vision-Free",
        description="The name of the Together model to use",
    )
    temperature: float = Field(default=0.9, description="Sampling temperature to use")
    top_p: float = Field(default=0.7, description="Top p sampling parameter")
    top_k: int = Field(default=50, description="Top k sampling parameter")
    repetition_penalty: float = Field(
        default=1.0, description="Repetition penalty parameter"
    )

    _client: Together = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = Together()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        image_url: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Args:
            prompt: The text prompt to generate from.
            stop: Stop words (not supported in this implementation).
            run_manager: Callback manager for the run.
            image_url: Optional URL to an image to analyze.
            **kwargs: Additional keyword arguments passed to the Together API.

        Returns:
            The model output as a string.
        """
        if stop is not None:
            raise ValueError(
                "stop kwargs are not permitted for Together Vision models."
            )

        # Construct the message content
        content = [{"role": "user", "content": []}]

        # Add text prompt
        content[0]["content"].append({"type": "text", "text": prompt})

        # Add image if provided
        if image_url:
            content[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        # Make the API call
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=content,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            **kwargs,
        )

        return response.choices[0].message.content

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        image_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream is not currently supported for Together Vision models."""
        raise NotImplementedError(
            "Streaming is not currently supported for Together Vision models."
        )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "together_vision"


# vision_color_contrast_checker = TogetherVisionLLM(
#     model_name="meta-llama/Llama-Vision-Free",
#     temperature=1.0,
#     top_p=0.8,
#     top_k=10,
#     repetition_penalty=1.0
# )

# font_checker = TogetherVisionLLM(
#     model_name="meta-llama/Llama-Vision-Free",
#     temperature=0.8,
#     top_p=0.7,
#     top_k=25,
#     repetition_penalty=1.0
# )

# color_prompt = ChatPromptTemplate.from_template("""
#         You are a Expert Color Contrast Tester. Choosing the correct colors is really important
#         for any design to be accessible for a large number of people having various vision impairments.
#         Assume that the user is a newbie and has no prior knowledge of color contrast.

#         CONTEXT: {context}

#         TASK:
#         Your task is to return instances within the image wherever the above conditions are not met.
#         You have to return a JSON array of each instance describing its flaws with specific location of the flaw.

#         Return ONLY the JSON in the given format, do not return anything else.
#         If there are no issues just return an empty JSON.
#         """)

# with open('coloring_for_colorblindness.txt') as f:
#     r1 = vision_color_contrast_checker.invoke(
#         f"""You are a Expert Color Contrast Tester. Choosing the correct colors is really important
#         for any design to be accessible for a large number of people having various vision impairments.
#         Assume that the user is a newbie and has no prior knowledge of color contrast.
#         Start the output with JSON "{{"

#         CONTEXT:  {f.read()}
#         TASK:
#             Your task is to return instances within the image wherever the above conditions are not met. You have to return a JSON array of each instance describing its flaws with specific location of the flaw.
#             return ONLY the JSON in the given format, do not return anything else. If there are no issues just return an empty JSON. :
#         OUTPUT FORMAT:
#             {{
#             "Issue1": {{
#                 "Title": " ... ",
#                 "Description": " ... ",
#                 "Suggestion": " ... ",
#                 "Importance_score": <float value between 0.0 to 1.0>
#             }},
#             "Issue2": {{
#                 "Title": " ... ",
#                 "Description": " ... ",
#                 "Suggestion": " ... ",
#                 "Importance_score": <float value between 0.0 to 1.0>
#             }},
#             ...
#             }}
#     """,
#         image_url="https://cdn.discordapp.com/attachments/699317920022397019/1302381353504997428/Summer-Flyer-and-Poster.png?ex=6727e8a7&is=67269727&hm=57ec1e6d9b8098efc7e78885c78f1a4cbafdbee9a46f651427568ea6872def93&"
#     )
# print(r1)

# with open('font_for_vision.txt', encoding='utf8') as f:
#     r2 = font_checker.invoke(
#         f"""You are a Expert Font Style Evaluater. Using information in the context provided, make suggestions
#              for the design to be accessible for a large number of people having various vision impairments. start the output with "JSON:"
#         CONTEXT:
#         {f.read()}

#         TASK:
#             Your task is to return instances within the image wherever the above conditions are not met. You have to return a JSON array of each instance describing its flaws with specific location of the flaw.
#             return ONLY the JSON in the given format, do not return anything else. If there are no issues just return an empty JSON. :
#         OUTPUT FORMAT:
#             {{
#             "Issue1": {{
#                 "Title": " ... ",
#                 "Description": " ... ",
#                 "Suggestion": " ... ",
#                 "Importance_score": <float value between 0.0 to 1.0>
#             }},
#             "Issue2": {{
#                 "Title": " ... ",
#                 "Description": " ... ",
#                 "Suggestion": " ... ",
#                 "Importance_score": <float value between 0.0 to 1.0>
#             }},
#             ...
#             }}
#     """,
#         image_url="https://cdn.discordapp.com/attachments/699317920022397019/1302381353504997428/Summer-Flyer-and-Poster.png?ex=6727e8a7&is=67269727&hm=57ec1e6d9b8098efc7e78885c78f1a4cbafdbee9a46f651427568ea6872def93&"
#     )
# print(r2)
# designleader = ChatTogether(
#     # together_api_key="YOUR_API_KEY",
#     model="meta-llama/Llama-Vision-Free",
# )

# # stream the response back from the model
# chat.invoke("Combine output of both the models and provide final feedback to the junior designer as a JSON file")


# %%
def create_accessibility_chain():
    """Create the main accessibility checking chain."""

    # Initialize models
    color_model = TogetherVisionLLM(
        # model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", temperature=1.0, top_p=0.8, top_k=10
        model_name="meta-llama/Llama-Vision-Free",
        temperature=1.0,
        top_p=0.8,
        top_k=10,
    )

    font_model = TogetherVisionLLM(
        model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        temperature=0.8,
        top_p=0.7,
        top_k=25,
        # model_name="meta-llama/Llama-Vision-Free", temperature=0.8, top_p=0.7, top_k=25
    )

    final_model = TogetherVisionLLM(
        # model_name="Qwen/Qwen2.5-72B-Instruct-Turbo", temperature=0.7, top_p=0.7, top_k=50
        model_name="meta-llama/Llama-Vision-Free",
        temperature=0.7,
        top_p=0.7,
        top_k=50,
    )

    # Create prompts
    color_prompt = PromptTemplate.from_template(
        """You are a Expert Color Contrast Tester. Choosing the correct colors is really important
        for any design to be accessible for a large number of people having various vision impairments.
        Assume that the user is a newbie and has no prior knowledge of color contrast.

        CONTEXT:  {context}
        TASK:
            Your task is to return instances within the image wherever the above conditions are not met. You have to return a JSON array of each instance describing its flaws with specific location of the flaw.
            return ONLY the data in the given format, do not return anything else. If there are no issues just return an EOL Character. :
        OUTPUT FORMAT:
  
            1.  "Title": " ... ",
                "Description": " ... ",
                "Suggestion": " ... ",
                "Importance_score": <float value between 0.0 to 1.0>
            2.  "Title": " ... ",
            ...
    """
    )

    font_prompt = PromptTemplate.from_template(
        """You are a Expert Font Style Evaluater. Using information in the context provided, make suggestions
             for the design to be accessible for a large number of people having various vision impairments
        CONTEXT:
        {context}

        TASK:
            Your task is to return instances within the image wherever the above conditions are not met. You have to return a JSON array of each instance describing its flaws with specific location of the flaw.
            return ONLY the data in the given format, do not return anything else. If there are no issues just return an EOL Character. :
        OUTPUT FORMAT:
  
            1.  "Title": " ... ",
                "Description": " ... ",
                "Suggestion": " ... ",
                "Importance_score": <float value between 0.0 to 1.0>
            2.  "Title": " ... ",
            ...
    """
    )

    final_prompt = PromptTemplate.from_template(
        """ You are the Design Leader, you have been provided the opinions of two experts on the design. 
        TASK:
        Aggregate output of both the Experts and sort the results based on their importance for creating an accesible design . Provide final feedback to the junior designer as a JSON file
        
        OUTPUT FORMAT:
        {{
            "Issue1": {{
                "Title": "...",
                "Description": "...",
                "Suggestion": "...",
                "Importance_score": <float between 0.0 to 1.0>
            }},
            ...
        }}

        Expert1 opinion: {color_result}

        Expert2 opinion: {font_result}
        """
    )

    # Create output parser
    output_parser = JsonOutputParser(pydantic_object=AccessibilityResponse)

    # Create individual chains
    color_chain = (
        {"context": RunnablePassthrough(), "image_url": RunnablePassthrough()}
        | color_prompt
        | color_model
    )

    font_chain = (
        {"context": RunnablePassthrough(), "image_url": RunnablePassthrough()}
        | font_prompt
        | font_model
    )

    # Create the final chain
    final_chain = (
        {"color_result": color_chain, "font_result": font_chain}
        | final_prompt
        | final_model
        | output_parser
    )

    return final_chain


# %%
def main(image_url):
    # Initialize the chain
    accessibility_chain = create_accessibility_chain()

    # Load contexts
    color_context = load_context("..\\knowledgebase\\coloring_for_colorblindness.txt")
    font_context = load_context("..\\knowledgebase\\font_for_vision.txt")

    # Image URL
    # image_url = "https://cdn.discordapp.com/attachments/699317920022397019/1302381353504997428/Summer-Flyer-and-Poster.png"

    # Run the chain
    result = accessibility_chain.invoke(
        {
            "context": {"color_context": color_context, "font_context": font_context},
            "image_url": image_url,
        }
    )

    # Print results
    return json.dumps(result, indent=2)


# %%
# main()

# %%
# import weaviate
# from weaviate.classes.init import Auth

# %%
# wcd_url = os.environ["WCD_URL"]
# wcd_api_key = os.environ["WCD_API_KEY"]

# weaviate_client = weaviate.connect_to_weaviate_cloud(
#     cluster_url=wcd_url,                                    # Replace with your Weaviate Cloud URL
#     auth_credentials=Auth.api_key(wcd_api_key),             # Replace with your Weaviate Cloud key
# )
# from langchain_together import TogetherEmbeddings

# embeddings = TogetherEmbeddings(
#     model="togethercomputer/m2-bert-80M-32k-retrieval",
# )
# from langchain_together import ChatTogether
# from langchain_weaviate.vectorstores import WeaviateVectorStore
# loader = TextLoader("./accessibility_posters_gilson2007.txt", encoding='utf8')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1300, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)
# db = WeaviateVectorStore.from_documents(docs, embeddings, client=weaviate_client)
# loader = TextLoader("coloring_for_colorblindness.txt", encoding='utf8')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1300, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)
# db = WeaviateVectorStore.from_documents(docs, embeddings, client=weaviate_client)
