from openai import AsyncOpenAI
from tiktoken import get_encoding
from haystack import (
    BaseNeedleHaystackTester,
    NeedleHaystackPlotter,
    StackedNeedleHaystackTester,
)
from dataclasses import asdict
import json
from pathlib import Path
import asyncio
from transformers import AutoTokenizer


class OpenAINeedleHaystackTester(BaseNeedleHaystackTester):
    def __init__(
        self,
        context_length: int,
        bench_ai: AsyncOpenAI,
        bench_ai_lock: asyncio.Lock,
        bench_ai_name: str,
        eval_ai: AsyncOpenAI,
        eval_ai_lock: asyncio.Lock,
        eval_ai_name: str,
        openai_tokenizer: str | None,
        transformers_tokenizer: str | None,
    ):
        super().__init__(context_length, 300)
        self.bench_ai = bench_ai
        self.bench_ai_lock = bench_ai_lock
        self.bench_ai_name = bench_ai_name
        self.eval_ai = eval_ai
        self.eval_ai_lock = eval_ai_lock
        self.eval_ai_name = eval_ai_name
        self.openai_tokenizer = (
            get_encoding(openai_tokenizer) if openai_tokenizer else None
        )
        self.transformers_tokenizer = (
            AutoTokenizer.from_pretrained(transformers_tokenizer)
            if transformers_tokenizer
            else None
        )
        if self.openai_tokenizer is None and self.transformers_tokenizer is None:
            raise ValueError("At least one tokenizer must be provided")
        if (
            self.openai_tokenizer is not None
            and self.transformers_tokenizer is not None
        ):
            print("Both tokenizers provided, prefering OpenAI tokenizer")

    async def answer_question(self, context: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct",
            },
            {"role": "user", "content": context},
            {
                "role": "user",
                "content": f"{question} Don't give information outside the document or repeat your findings",
            },
        ]
        async with self.bench_ai_lock:
            completion = await self.bench_ai.chat.completions.create(
                model=self.bench_ai_name,
                messages=messages,
            )
        return completion.choices[0].message.content.strip()

    async def evaluate_response(
        self, model_response: str, true_answer: str, question_asked: str
    ) -> int:
        prompt = f"""
        You are a helpful AI bot that evaluates answers based on their accuracy to a reference answer.
        Score 1: The answer is completely unrelated to the reference.
        Score 3: The answer has minor relevance but does not align with the reference.
        Score 5: The answer has moderate relevance but contains inaccuracies.
        Score 7: The answer aligns with the reference but has minor omissions.
        Score 10: The answer is completely accurate and aligns perfectly with the reference.
        Only respond with a numerical score.

        Question Asked: {question_asked}
        True Answer: {true_answer}
        Model's Response: {model_response}

        What score would you give the model's response based on its accuracy to the true answer?
        """
        async with self.eval_ai_lock:
            completion = await self.eval_ai.chat.completions.create(
                model=self.eval_ai_name,
                messages=[{"role": "user", "content": prompt}],
            )
        score_str = completion.choices[0].message.content.strip()
        try:
            score = int(score_str)
            if score < 1 or score > 10:
                raise ValueError
            return score
        except ValueError:
            raise ValueError(f"Invalid score received from model: {score_str}")

    def encode_tokens(self, text: str) -> list[int]:
        if self.openai_tokenizer is not None:
            return self.openai_tokenizer.encode(text)
        else:
            return self.transformers_tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int]) -> str:
        if self.openai_tokenizer is not None:
            return self.openai_tokenizer.decode(tokens)
        else:
            return self.transformers_tokenizer.decode(tokens)


async def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--bench-openai-api-key",
        type=str,
        default="hello",
        help="API key for the benchmarked model",
    )
    parser.add_argument(
        "--bench-openai-api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="API base URL for the benchmarked model",
    )
    parser.add_argument(
        "--bench-openai-model",
        type=str,
        default="gpt-oss-20b",
        help="Model name for the benchmarked model",
    )
    parser.add_argument(
        "--bench-concurrent",
        type=int,
        default=20,
        help="Number of concurrent requests to the benchmarked model",
    )
    parser.add_argument(
        "--eval-openai-api-base",
        type=str,
        default=None,
        help="API base URL for the evaluation model, defaults to the benchmarked model's API base URL if not provided",
    )
    parser.add_argument(
        "--eval-openai-api-key",
        type=str,
        default=None,
        help="API key for the evaluation model, defaults to the benchmarked model's API key if not provided",
    )
    parser.add_argument(
        "--eval-openai-model",
        type=str,
        default=None,
        help="Model name for the evaluation model, defaults to the benchmarked model's name if not provided",
    )
    parser.add_argument(
        "--eval-concurrent",
        type=int,
        default=None,
        help="Number of concurrent requests to the evaluation model, defaults to the benchmarked model's concurrency if not provided",
    )
    parser.add_argument(
        "--openai-tokenizer",
        type=str,
        default="o200k_harmony",
        help="OpenAI tokenizer name, e.g., 'o200k_harmony'",
    )
    parser.add_argument(
        "--transformers-tokenizer",
        type=str,
        default=None,
        help="Transformers tokenizer name, e.g., 'meta-llama/Llama-3.2-1B-Instruct'",
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="1024,2048,4096,8192,12288,16384,20480,24576,28672,32768",
        help="Comma-separated list of context lengths to test",
    )
    parser.add_argument(
        "--context-file-dir",
        type=str,
        default="./paul_graham_essays",
        help="Directory containing text files to use as context",
    )
    parser.add_argument(
        "--tick",
        type=int,
        default=5,
        help="Number of increments to divide the context length into",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is the best thing to do in San Francisco?",
        help="The question to ask the model",
    )
    parser.add_argument(
        "--needle",
        type=str,
        default="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        help="The needle (true answer) to look for in the context",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=3,
        help="Number of times to repeat each test for averaging",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the output plot, defaults to 'Benchmark result for {bench_openai_model}'",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="results.png",
        help="Path to save the output plot image, if not provided the plot will not be saved",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Path to save the output results in JSONL format, if not provided the results will not be saved",
    )
    args = parser.parse_args()
    if args.eval_openai_api_base is None:
        args.eval_openai_api_base = args.bench_openai_api_base
    if args.eval_openai_api_key is None:
        args.eval_openai_api_key = args.bench_openai_api_key
    if args.eval_openai_model is None:
        args.eval_openai_model = args.bench_openai_model
    if args.eval_concurrent is None:
        args.eval_concurrent = args.bench_concurrent
    if args.title is None:
        args.title = f"Benchmark result for {args.bench_openai_model}"
    bench_ai = AsyncOpenAI(
        base_url=args.bench_openai_api_base, api_key=args.bench_openai_api_key
    )
    bench_ai_lock = asyncio.Semaphore(args.bench_concurrent)
    eval_ai = AsyncOpenAI(
        base_url=args.eval_openai_api_base, api_key=args.eval_openai_api_key
    )
    eval_ai_lock = asyncio.Semaphore(args.eval_concurrent)

    stack = StackedNeedleHaystackTester(
        lambda ctx: OpenAINeedleHaystackTester(
            ctx,
            bench_ai,
            bench_ai_lock,
            args.bench_openai_model,
            eval_ai,
            eval_ai_lock,
            args.eval_openai_model,
            args.openai_tokenizer,
            args.transformers_tokenizer,
        )
    )
    contexts = []
    for file in Path(args.context_file_dir).glob("*.txt"):
        with open(file, "r") as f:
            contexts.append(f.read())
    res = await stack.run(
        context_lengths=[int(x) for x in args.context_lengths.split(",")],
        tick=args.tick,
        contexts=contexts,
        question=args.question,
        needle=args.needle,
        times=args.times,
    )
    if args.output_image is not None:
        plotter = NeedleHaystackPlotter()
        plt = plotter.plot_scores(res, args.title)
        plt.savefig(args.output_image)
    if args.output_jsonl is not None:
        with open(args.output_jsonl, "w") as f:
            for r in res:
                f.write(json.dumps(asdict(r)) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
