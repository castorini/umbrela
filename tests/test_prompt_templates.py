import os
import tempfile
import unittest

import pytest

from umbrela.cli.prompt_view import build_rendered_prompt_view
from umbrela.prompts import get_prompt_template, render_prompts

pytestmark = pytest.mark.core


EXPECTED_ZERO_BASIC = """You are an expert judge of a content. Using your internal knowledge and simple commonsense reasoning, try to verify if the passage is relevance category to the query.
Here, "0" represent that the passage has nothing to do with the query, "1" represents that the passage seems related to the query but does not answer it, "2" represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and "3" represents that the passage is dedicated to the query and contains the exact answer.

Provide explanation for the relevance and give your answer with from one of the categories 0, 1, 2 or 3 only. One of the categorical values if compulsory in answer.

Instructions: Think about the question. After explaining your reasoning, provide your answer in terms of 0, 1, 2 or 3 category. Only provide the relevance category on the last line. Do not provide any further details on the last line.

###

Query: {query}
Passage: {passage}

Explanation:"""

EXPECTED_ZERO_BING = """Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query, 
1 = represents that the passage seems related to the query but does not answer it, 
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and 
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: {query}
Passage: {passage}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning."""

EXPECTED_FEWSHOT_BASIC = """You are an expert judge of a content. Using your internal knowledge and simple commonsense reasoning, try to verify if the passage is relevance category to the query.
Here, "0" represent that the passage has nothing to do with the query, "1" represents that the passage seems related to the query but does not answer it, "2" represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and "3" represents that the passage is dedicated to the query and contains the exact answer.

Following are some of the examples of relevance categorizations for different categories:

{examples}

Provide explanation for the relevance and give your answer with from one of the categories 0, 1, 2 or 3 only. One of the categorical values if compulsory in answer.

Instructions: Think about the question. After explaining your reasoning, provide your answer in terms of 0, 1, 2 or 3 category. Only provide the relevance category on the last line. Do not provide any further details on the last line.

###

Query: {query}
Passage: {passage}

Explanation:"""

EXPECTED_FEWSHOT_BING = """Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query, 
1 = represents that the passage seems related to the query but does not answer it, 
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and 
3 = represents that the passage is dedicated to the query and contains the exact answer.

Following are some of the examples of relevance categorizations for different categories:

###

query: anthropological definition of environment
passage: Abiotic factors are all of the non-living things in an environment. All biotic factors in an environment are dependent upon the abiotic factors in an environment. Some examples of biotic factors in the tropical rainforest are toucans, frogs, snakes, and lizards.Abiotic factors in the tropical rainforest include humidity, soil composition, temperature, and sunlight.very environment contains what are called 'biotic' and 'abiotic' factors. In this lesson you will learn the definition and importance of abiotic factors. You will also view some examples of abiotic factors that are present in the tropical rainforest.

### Steps:

1. **Consider the underlying intent of the search:**
   - The intent is to find the anthropological definition of "environment," which implies looking for a definition within the context of anthropology.

2. **Measure how well the content matches a likely intent of the query (M):**
   - The passage discusses abiotic and biotic factors in an environment, specifically in a tropical rainforest, but does not relate these concepts to anthropology or provide a definition within an anthropological context.

3. **Measure how trustworthy the passage is (T):**
   - The trustworthiness of the source cannot be determined from the passage alone.

4. **Consider the aspects above and the relative importance of each, and decide on a final score (O):**
   - The passage does not directly address the anthropological definition of "environment" and is more focused on ecological concepts. It does not provide relevant information for the query.

##final score: 0

###

query: anthropological definition of environment
passage: Definition of Environment. Environment: The sum of the total of the elements, factors and conditions in the surroundings which may have an impact on the development, action or survival of an organism or group of organisms. Search MedTerms:

### Steps:
1. **Consider the underlying intent of the search:**
   - The intent is to find the anthropological definition of "environment," which implies looking for a definition that is specifically framed within the context of anthropology.

2. **Measure how well the content matches a likely intent of the query (M):**
   - The passage provides a general definition of "environment" but does not specifically address the anthropological context.

3. **Measure how trustworthy the passage is (T):**
   - The trustworthiness of the source cannot be determined from the passage alone. It mentions "Search MedTerms," which implies a medical terminology source that may not be directly relevant to anthropology.

4. **Consider the aspects above and the relative importance of each, and decide on a final score (O):**
   - The passage is somewhat related to the query but does not meet the specific intent (anthropological context) and the trustworthiness is ambiguous.

##final score: 1

###

query: anthropological definition of environment
passage: Graduate Study in Anthropology. The graduate program in biological anthropology at CU Boulder offers training in several areas, including primatology, human biology, and paleoanthropology. We share an interest in human ecology, the broad integrative area of anthropology that focuses on the interactions of culture, biology and the environment.

### Steps:

1. **Consider the underlying intent of the search:**
   - The intent is to find the anthropological definition of "environment," implying a focus on how anthropology defines and interprets the environment within its context.

2. **Measure how well the content matches a likely intent of the query (M):**
   - The passage discusses human ecology, a broad area of anthropology that examines the interactions between culture, biology, and the environment. This aligns well with the anthropological context of the environment.

3. **Measure how trustworthy the passage is (T):**
   - The source refers to a graduate program at CU Boulder, a reputable institution, which indicates a high level of trustworthiness.

4. **Consider the aspects above and the relative importance of each, and decide on a final score (O):**
   - The passage is relevant and provides a contextual understanding of the environment in anthropology, though it does not offer a precise definition. This makes it somewhat detailed and relevant, thus scoring higher.

##final score: 2

###

query: anthropological definition of environment
passage: Archaeology, which studies past human cultures through investigation of physical evidence, is thought of as a branch of anthropology in the United States, although in Europe, it is viewed as a discipline in its own right, or related to other disciplines.nvironmental anthropology is a sub-specialty within the field of anthropology that takes an active role in examining the relationships between humans and their environment across space and time.

### Steps:

1. **Consider the underlying intent of the search:**
   - The intent is to find the anthropological definition of "environment," which means looking for an explanation of how anthropology views and studies the environment.

2. **Measure how well the content matches a likely intent of the query (M):**
   - The passage explicitly mentions "environmental anthropology," a sub-specialty that examines the relationships between humans and their environment, directly addressing the anthropological perspective on the environment.

3. **Measure how trustworthy the passage is (T):**
   - The passage appears to provide an academic and structured explanation of anthropology sub-fields, indicating a trustworthy source, likely from an educational or scholarly background.

4. **Consider the aspects above and the relative importance of each, and decide on a final score (O):**
   - The passage is directly relevant to the query, providing specific information about the field of environmental anthropology and its focus on human-environment relationships, making it highly pertinent.

##final score: 3

###

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: {query}
Passage: {passage}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning."""


class PromptTemplateTests(unittest.TestCase):
    def test_builtin_templates_preserve_exact_surface(self) -> None:
        cases = [
            ("basic", 0, EXPECTED_ZERO_BASIC),
            ("bing", 0, EXPECTED_ZERO_BING),
            ("basic", 2, EXPECTED_FEWSHOT_BASIC),
            ("bing", 2, EXPECTED_FEWSHOT_BING),
        ]

        for prompt_type, few_shot_count, expected in cases:
            with self.subTest(prompt_type=prompt_type, few_shot_count=few_shot_count):
                template = get_prompt_template(
                    prompt_file=None,
                    prompt_type=prompt_type,
                    few_shot_count=few_shot_count,
                )
                self.assertEqual(template.prefix_user, expected)

    def test_render_prompts_preserves_flat_prompt_surface(self) -> None:
        template = get_prompt_template(
            prompt_file=None,
            prompt_type="basic",
            few_shot_count=2,
        )
        prompts = render_prompts(
            template,
            [("sample query", "sample passage")],
            "demo examples",
        )

        self.assertEqual(
            prompts,
            [
                EXPECTED_FEWSHOT_BASIC.format(
                    examples="demo examples",
                    query="sample query",
                    passage="sample passage",
                )
            ],
        )

    def test_custom_yaml_template_must_include_required_placeholders(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(
                'method: "custom"\nsystem_message: ""\nprefix_user: "Query: {query}"\n'
            )
            template_path = handle.name

        self.addCleanup(lambda: os.unlink(template_path))

        with self.assertRaisesRegex(
            ValueError,
            r"Prompt template must provide the fields `\{examples\}`, `\{passage\}`\.",
        ):
            get_prompt_template(
                prompt_file=template_path,
                prompt_type=None,
                few_shot_count=0,
            )

    def test_custom_yaml_template_renders_like_builtin_path(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(
                'method: "custom"\n'
                'system_message: ""\n'
                'prefix_user: "{examples}\\nQuery: {query}\\nPassage: {passage}\\n"\n'
            )
            template_path = handle.name

        self.addCleanup(lambda: os.unlink(template_path))

        template = get_prompt_template(
            prompt_file=template_path,
            prompt_type=None,
            few_shot_count=0,
        )
        self.assertEqual(
            template.render(
                examples="sample examples",
                query="sample query",
                passage="sample passage",
            ),
            "sample examples\nQuery: sample query\nPassage: sample passage\n",
        )

    def test_template_metadata_reports_source_parts_and_placeholders(self) -> None:
        template = get_prompt_template(
            prompt_file=None,
            prompt_type="basic",
            few_shot_count=2,
        )

        self.assertEqual(template.method, "qrel_fewshot_basic")
        self.assertIn("prompt_templates/qrel_fewshot_basic.yaml", template.source_path)
        self.assertEqual(
            template.placeholders,
            ("examples", "query", "passage"),
        )
        self.assertEqual(
            template.raw_parts(),
            {
                "system_message": "",
                "prefix_user": EXPECTED_FEWSHOT_BASIC,
            },
        )
        self.assertEqual(
            template.metadata(),
            {
                "method": "qrel_fewshot_basic",
                "source_path": template.source_path,
                "system_message": "",
                "prefix_user": EXPECTED_FEWSHOT_BASIC,
                "placeholders": ["examples", "query", "passage"],
            },
        )

    def test_custom_template_metadata_reports_custom_source_and_parts(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(
                'method: "custom"\n'
                'system_message: "system text"\n'
                'prefix_user: "Examples: {examples}\\nQuery: {query}\\nPassage: {passage}"\n'
            )
            template_path = handle.name

        self.addCleanup(lambda: os.unlink(template_path))

        template = get_prompt_template(
            prompt_file=template_path,
            prompt_type=None,
            few_shot_count=0,
        )

        self.assertEqual(template.placeholders, ("examples", "query", "passage"))
        self.assertEqual(
            template.metadata(),
            {
                "method": "custom",
                "source_path": template.source_path,
                "system_message": "system text",
                "prefix_user": "Examples: {examples}\nQuery: {query}\nPassage: {passage}",
                "placeholders": ["examples", "query", "passage"],
            },
        )

    def test_rendered_prompt_view_preserves_selector_metadata(self) -> None:
        template = get_prompt_template(
            prompt_file=None,
            prompt_type="basic",
            few_shot_count=2,
        )

        view = build_rendered_prompt_view(
            template,
            prompt_file=None,
            prompt_type="basic",
            few_shot_count=2,
            candidate_index=0,
            query="sample query",
            passage="sample passage",
            examples="demo examples",
        )

        self.assertEqual(
            view["selector"],
            {
                "prompt_file": None,
                "prompt_type": "basic",
                "few_shot_count": 2,
                "candidate_index": 0,
            },
        )
