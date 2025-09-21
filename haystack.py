from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from typing import Callable
from dataclasses import dataclass
import asyncio
from tqdm.asyncio import tqdm


@dataclass
class NeedleHaystackResult:
    context_length: int
    document_depth: float
    score: float
    response: list[str]


class BaseNeedleHaystackTester:
    def __init__(self, context_length: int, answer_length: int):
        self.question_context_length = context_length - answer_length
        self.context_length = context_length

    def insert_needles(self, needle: str, context: str, depth_percent: float):
        context_tokens = self.encode_tokens(context)
        needle_tokens = self.encode_tokens(needle)
        if len(context_tokens) + len(needle_tokens) > self.question_context_length:
            context_tokens = context_tokens[
                : self.question_context_length - len(needle_tokens)
            ]
        if depth_percent == 100:
            context_tokens = context_tokens + needle_tokens
        else:
            insertion_point = int(len(context_tokens) * (depth_percent / 100))
            tokens_new_context = context_tokens[:insertion_point]
            period_tokens = self.encode_tokens(".")
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = context_tokens[:insertion_point]
            context_tokens = (
                context_tokens[:insertion_point]
                + needle_tokens
                + context_tokens[insertion_point:]
            )

        return self.decode_tokens(context_tokens)

    async def run_percent(
        self,
        context: str,
        question: str,
        needle: str,
        depth_percent: float,
        times: int,
    ):
        context = self.insert_needles(needle, context, depth_percent)

        async def get_response_score():
            response = await self.answer_question(context, question)
            score = await self.evaluate_response(response, needle, question)
            return response, score

        responses_scores = await asyncio.gather(
            *[get_response_score() for _ in range(times)]
        )
        avg_score = sum(score for _, score in responses_scores) // times
        response = [response for response, _ in responses_scores]
        return NeedleHaystackResult(
            context_length=self.context_length,
            document_depth=depth_percent,
            score=avg_score,
            response=response,
        )

    async def answer_question(self, context: str, question: str) -> str:
        raise NotImplementedError

    async def evaluate_response(
        self, model_response: str, true_answer: str, question_asked: str
    ) -> int:
        raise NotImplementedError

    def encode_tokens(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode_tokens(self, tokens: list[int]) -> str:
        raise NotImplementedError


class StackedNeedleHaystackTester:
    def __init__(self, tester_cls_factory: Callable[[int], BaseNeedleHaystackTester]):
        self.tester_cls_factory = tester_cls_factory

    @staticmethod
    def take_context(tester: BaseNeedleHaystackTester, contexts: list[str]):
        context = ""
        while len(tester.encode_tokens(context)) < tester.question_context_length:
            for c in contexts:
                context += c
        return context

    @staticmethod
    async def safe_run(coro):
        try:
            return await coro
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    async def run(
        self,
        context_lengths: list[int],
        tick: int,
        contexts: list[str],
        question: str,
        needle: str,
        times: int,
    ):
        tasks = []
        for context_length in context_lengths:
            ev = self.tester_cls_factory(context_length)
            context = StackedNeedleHaystackTester.take_context(ev, contexts)
            for i in range(tick):
                tasks.append(
                    StackedNeedleHaystackTester.safe_run(
                        ev.run_percent(
                            context,
                            question,
                            needle,
                            100.0 / tick * (i + 1),
                            times,
                        )
                    )
                )
        return [r for r in await tqdm.gather(*tasks) if r is not None]


class NeedleHaystackPlotter:
    def __init__(
        self,
        document_depth_name="Document Depth",
        context_length_name="Context Length",
        score_name="Score",
    ):
        self.document_depth_name = document_depth_name
        self.context_length_name = context_length_name
        self.score_name = score_name

    def plot_scores(self, datas: list[NeedleHaystackResult], title: str):
        df = pd.DataFrame(
            [
                {
                    self.document_depth_name: data.document_depth,
                    self.context_length_name: data.context_length,
                    self.score_name: data.score,
                }
                for data in datas
            ]
        )
        pivot_table = pd.pivot_table(
            df,
            values=self.score_name,
            index=[self.document_depth_name, self.context_length_name],
            aggfunc="mean",
        ).reset_index()
        pivot_table = pivot_table.pivot(
            index=self.document_depth_name,
            columns=self.context_length_name,
            values=self.score_name,
        )
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
        )
        plt.figure(figsize=(17.5, 8))
        sns.heatmap(
            pivot_table,
            fmt="g",
            cmap=cmap,
            cbar_kws={"label": self.score_name},
            vmin=0,
            vmax=10,
        )
        plt.title(title)
        plt.xlabel("Token Limit")
        plt.ylabel("Depth Percent")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        return plt
