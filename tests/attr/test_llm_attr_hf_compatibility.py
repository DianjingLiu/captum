#!/usr/bin/env python3

# pyre-strict

from typing import cast, Dict, Optional, Type

import torch
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.llm_attr import LLMAttribution
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr._utils.interpretable_input import TextTemplateInput
from parameterized import parameterized, parameterized_class
from tests.helpers import BaseTest
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

@parameterized_class(
    ("device", "use_cached_outputs"),
    (
        [("cpu", True), ("cpu", False), ("cuda", True), ("cuda", False)]
        if torch.cuda.is_available()
        else [("cpu", True), ("cpu", False)]
    ),
)
class TestLLMAttr(BaseTest):
    device: str
    use_cached_outputs: bool

    # pyre-fixme[56]: Pyre was not able to infer the type of argument `comprehension
    @parameterized.expand(
        [
            (
                AttrClass,
                delta,
                n_samples,
            )
            for AttrClass, delta, n_samples in zip(
                (FeatureAblation, ShapleyValueSampling, ShapleyValues),  # AttrClass
                (0.001, 0.001, 0.001),  # delta
                (None, 1000, None),  # n_samples
            )
        ]
    )
    def test_llm_attr(
        self,
        AttrClass: Type[PerturbationAttribution],
        delta: float,
        n_samples: Optional[int],
    ) -> None:
        attr_kws: Dict[str, int] = {}
        if n_samples is not None:
            attr_kws["n_samples"] = n_samples

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        llm = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        
        llm.to(self.device)
        llm.eval()
        llm_attr = LLMAttribution(AttrClass(llm), tokenizer)

        inp = TextTemplateInput("{} b {} {} e {}", ["a", "c", "d", "f"])
        res = llm_attr.attribute(
            inp,
            "m n o p q",
            use_cached_outputs=self.use_cached_outputs,
            # pyre-fixme[6]: In call `LLMAttribution.attribute`,
            # for 4th positional argument, expected
            # `Optional[typing.Callable[..., typing.Any]]` but got `int`.
            **attr_kws,  # type: ignore
        )
        self.assertEqual(res.seq_attr.shape, (4,))
        self.assertEqual(cast(Tensor, res.token_attr).shape, (5, 4))
        self.assertEqual(res.input_tokens, ["a", "c", "d", "f"])
        self.assertEqual(len(res.output_tokens), 5)
        self.assertEqual(res.seq_attr.device.type, self.device)
        self.assertEqual(cast(Tensor, res.token_attr).device.type, self.device)
