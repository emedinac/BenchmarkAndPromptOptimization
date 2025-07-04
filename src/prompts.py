import numpy as np
from prepare_dataset import get_dataset_embeddings, SentenceTransformer
import ollama_llms as llms
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import copy

# Tried with 5-10 few-shot examples just to validate the robustness of the LLM.
simple_examples_CoT = [{"Q": "Describe the process of a stock split and its impact on shareholders.",
                        "A": "1. A stock split involves a company issuing additional shares to shareholders, increasing the total number of shares outstanding. 2. The share price is adjusted accordingly, so the overall value of a shareholder's investment remains unchanged. 3. For example, in a 2-for-1 split, shareholders receive an additional share for each they already own, and the share price is halved. 4. The primary purpose is to make shares more affordable for investors. 5. While the split doesn't affect the company's market capitalization, it can increase liquidity and attract more retail investors."},
                       {"Q": "How does the Federal Reserve's interest rate affect stock markets?",
                        "A": "1. The Federal Reserve sets the federal funds rate, influencing borrowing costs for banks. 2. Lower interest rates reduce borrowing costs, encouraging businesses to invest and consumers to spend. 3. Increased spending can lead to higher corporate earnings, boosting stock prices. 4. Conversely, higher interest rates can dampen economic activity, leading to lower earnings expectations and declining stock prices. 5. Therefore, the Fed's interest rate decisions play a crucial role in shaping stock market performance."},
                       {"Q": "Explain the concept of market capitalization and its significance.",
                        "A": "1. Market capitalization is calculated by multiplying a company's share price by its total number of outstanding shares. 2. It represents the total value of a company's equity. 3. Companies are often categorized based on their market cap: small-cap, mid-cap, and large-cap. 4. A higher market cap can indicate a more established company with stable earnings. 5. Investors use market cap to assess a company's size, risk, and growth potential."},
                       {"Q": "What factors contribute to a company's price-to-earnings (P/E) ratio?",
                        "A": "1. The P/E ratio is calculated by dividing the current share price by the company's earnings per share (EPS). 2. A high P/E ratio may indicate that investors expect future growth, while a low P/E ratio could suggest undervaluation or potential issues. 3. Industry norms and comparisons with peers are important for context. 4. A company's growth prospects, risk profile, and historical performance also influence its P/E ratio. 5. Therefore, the P/E ratio serves as a tool for evaluating a company's valuation relative to its earnings."},
                       {"Q": "How do dividends impact stock prices?",
                        "A": "1. When a company announces a dividend, its stock price typically drops by the dividend amount on the ex-dividend date. 2. This adjustment reflects the outflow of cash from the company to shareholders. 3. Regular dividend payments can signal financial stability and attract income-focused investors. 4. However, if a company cuts or eliminates its dividend, it may raise concerns about financial health, potentially leading to a decline in stock price. 5. Therefore, dividends can influence investor perception and stock price movements."},
                       {"Q": "What is the role of a stock exchange in financial markets?",
                        "A": "1. A stock exchange provides a platform for buying and selling securities, ensuring liquidity and price discovery. 2. It establishes rules and regulations to maintain fair trading practices. 3. Exchanges facilitate the listing of companies, allowing them to raise capital through public offerings. 4. They also provide transparency by requiring companies to disclose financial information. 5. Overall, stock exchanges are vital for the efficient functioning of financial markets."},
                       {"Q": "Describe the concept of diversification in investment portfolios.",
                        "A": "1. Diversification involves spreading investments across various assets to reduce risk. 2. By holding a mix of asset types, sectors, and geographic regions, investors can mitigate the impact of poor performance in any single investment. 3. The goal is to achieve a more stable overall return. 4. Diversification doesn't guarantee profits or protect against losses, but it can help manage risk. 5. Therefore, it's a fundamental strategy in portfolio management."},
                       {"Q": "How do geopolitical events affect financial markets?",
                        "A": "1. Geopolitical events, such as wars, elections, and trade negotiations, can introduce uncertainty into financial markets. 2. Markets may react to perceived risks, leading to increased volatility. 3. For instance, conflicts can disrupt supply chains, affecting commodity prices and corporate earnings. 4. Conversely, diplomatic resolutions can restore investor confidence. 5. Therefore, geopolitical events play a significant role in shaping market sentiment and performance."},
                       {"Q": "Explain the concept of liquidity in financial markets.",
                        "A": "1. Liquidity refers to the ease with which an asset can be bought or sold without affecting its price. 2. Highly liquid markets have many buyers and sellers, leading to tight bid-ask spreads and quick transactions. 3. Conversely, illiquid markets may experience price fluctuations due to fewer participants. 4. Liquidity is crucial for investors to enter or exit positions efficiently. 5. Therefore, it is a key consideration in market analysis and investment decisions."},
                       {"Q": "Explain step by step how a company's earnings per share (EPS) is calculated.",
                        "A": "1. EPS is calculated by dividing a company's net income by the number of outstanding shares. 2. Net income is found on the company's income statement. 3. The number of outstanding shares is typically reported in the company's quarterly or annual reports. 4. The formula is: EPS = Net Income / Outstanding Shares. 5. This metric indicates the portion of a company's profit allocated to each outstanding share of common stock."},
                       {"Q": "Describe the process of margin trading and its risks.",
                        "A": "1. Margin trading involves borrowing funds from a broker to trade financial assets. 2. The investor must open a margin account and deposit a minimum amount, known as the margin requirement. 3. The broker lends the investor additional funds to purchase more securities than they could with their own capital. 4. Risks include the potential for amplified losses, margin calls if the value of securities falls, and interest charges on borrowed funds."},
                       {"Q": "How does a company decide whether to issue debt or equity to raise capital?",
                        "A": "1. The decision depends on factors like the company's current debt levels, market conditions, and investor appetite. 2. Issuing debt involves borrowing funds that must be repaid with interest, while issuing equity involves selling ownership stakes in the company. 3. Debt is preferable when interest rates are low and the company has a stable cash flow. 4. Equity may be chosen when the company wants to avoid increasing debt levels or when market conditions favor high valuations."},
                       {"Q": "Explain the concept of a cryptocurrency wallet and its types.",
                        "A": "1. A cryptocurrency wallet is a digital tool that allows users to store and manage their cryptocurrencies. 2. There are two main types: hot wallets, which are connected to the internet, and cold wallets, which are offline. 3. Hot wallets are more convenient for frequent transactions but are more vulnerable to hacks. 4. Cold wallets offer enhanced security and are ideal for long-term storage. 5. Both types store private keys that are necessary to access and manage crypto assets."},
                       {"Q": "How do stock buybacks affect a company's stock price?",
                        "A": "1. A stock buyback occurs when a company repurchases its own shares from the market. 2. This reduces the number of outstanding shares, potentially increasing earnings per share (EPS). 3. The reduction in share supply can lead to an increase in the stock price if demand remains constant. 4. Buybacks can signal to investors that the company believes its stock is undervalued. 5. However, if done excessively, buybacks may divert funds from other investments and could be viewed negatively."},
                       ]
# Tried with 5-10 few-shot examples just to validate the robustness of the LLM.
simple_examples_few_shot = [{"Q": "What is the market capitalization of Apple?",
                             "A": "As of June 2025, Apple's market capitalization is approximately $3.1 trillion."},
                            {"Q": "What factors influence a company's market capitalization?",
                             "A": "Market capitalization is influenced by a company's stock price and the number of outstanding shares. Factors such as earnings reports, market sentiment, and industry trends can affect the stock price, thereby impacting the market cap."},
                            {"Q": "Which company has the highest market cap in the world?",
                             "A": "Microsoft holds the highest market capitalization globally, estimated at $3.496 trillion."},
                            {"Q": "What is the P/E ratio of GOOGLE?",
                             "A": "The P/E (Price-to-Earnings) ratio of Google (Alphabet Inc., GOOGL) is currently around 19.72. This is based on the latest stock price and the company's earnings per share (EPS) over the trailing twelve months (TTM)."},
                            {"Q": "How many shares of Tesla are outstanding?",
                             "A": "As of the latest reports, Tesla has approximately 3.2 billion shares outstanding. Today 3/Jul/2025"},
                            {"Q": "What is the P/E ratio of Amazon?",
                             "A": "The current price-to-earnings (P/E) ratio for Amazon (AMZN) is 35.41, as of June 26, 2025, according to Public.com. This ratio is slightly lower than its 12-month average of 41.97 and the average over the last three years of 47.96."},
                            {"Q": "Which cryptocurrency has the largest market cap?",
                             "A": "Bitcoin consistently holds the largest market capitalization among cryptocurrencies."},
                            {"Q": "What is the dividend yield of Microsoft?",
                             "A": "Ex-Dividend Date 05/15/2025 ; Dividend Yield 0.67% ; Annual Dividend $3.32 ; P/E Ratio 48.15."},
                            {"Q": "How many shares of Meta Platforms are outstanding?",
                             "A": "Meta Platforms has approximately 2.8 billion shares outstanding, according to the latest filings."},
                            {"Q": "What is the market capitalization of Ethereum?",
                             "A": "Ethereum is up 6.02% in the last 24 hours. The current CoinMarketCap ranking is #2, with a live market cap of $313,134,421,913 USD. Today 3/Jul/2025"},
                            # {"Q": "Which company is the largest semiconductor manufacturer by market cap?",
                            #  "A": "Taiwan Semiconductor Manufacturing Company (TSMC) is one of the largest semiconductor manufacturers by market capitalization."},
                            ]


# Here the input texts for experiments :)
test_texts = []

available_prompt_engineeing_tpye = ["auto-cot",
                                    "zero-cot",
                                    "few-shot-cot",
                                    "zero-shot",
                                    "few-shot",
                                    "self-consistency",
                                    ]


def faiss_kmeans(embeddings: np.ndarray, num_clusters: int = 10, n_iter: int = 25):
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, num_clusters, niter=n_iter, verbose=True)
    kmeans.train(embeddings)
    _, labels = kmeans.index.search(embeddings, 1)
    return labels.ravel(), kmeans.centroids


class Prompter:
    def __init__(self, beshort: bool = True, examples_to_use: int = None):
        '''
        inspired by reading: https://www.promptingguide.ai
        I avoided to use mix of methods to simplify the comparison.
        '''
        self.beshort = beshort
        self.select = examples_to_use  # select a number of cases for few-shot
        self.reference_dataset_embeddings = None
        self.reference_dataset = None
        self.centroids = None
        self.labels = None

    @staticmethod
    def _zero_shot(question):
        return question

    @staticmethod
    def _aggregate_few_shots(examples):
        return "\n".join(
            [f"Q: {ex['Q']}\nA: {ex['A']}" for ex in examples]
        )

    def _few_shot(self, question):
        if self.select is not None:
            examples = np.random.choice(simple_examples_few_shot, self.select)
        else:
            examples = simple_examples_few_shot
        few_shot_context = self._aggregate_few_shots(examples)
        return f"{few_shot_context}\nQ: {question}\nA:"

    def _few_shot_cot(self, question):
        if self.select is not None:
            examples = np.random.choice(simple_examples_CoT, self.select)
        else:
            examples = simple_examples_CoT
        few_shot_context = self._aggregate_few_shots(examples)
        return f"{few_shot_context}\nQ: {question}\nA:"

    @staticmethod
    def _zero_cot(question):
        return f"Question: {question}\nLet's think step-by-step. Answer:"

    def _compute_steps(self, exemplars):
        if self.reference_dataset_embeddings is None:
            self.reference_dataset_embeddings, self.reference_dataset = get_dataset_embeddings()

        if self.centroids is None:
            self.labels, self.centroids = faiss_kmeans(self.reference_dataset_embeddings,
                                                       num_clusters=exemplars,
                                                       n_iter=25)

            self.exemplar_indices = [int(np.argmin(np.linalg.norm(
                self.reference_dataset_embeddings - center, axis=1))) for center in self.centroids]
            self.exemplars = [self.reference_dataset[i] for i in self.exemplar_indices]

        # Generate chain-of-thought for each exemplar
        for ex in self.exemplars:
            ex_q = ex["Question"]
            cot_prompt = self._zero_cot(ex_q)
            cot_prompt = {'role': 'user',
                          'content': cot_prompt}
            ex_cot = llms.runLLM([cot_prompt])
            ex["Question"] = llms.remove_links(ex_q)  # temporal patch
            ex["cot"] = ex_cot
        return self.exemplars

    def _auto_cot(self, question, samples=7):
        # ref: https://github.com/amazon-science/auto-cot/blob/main/run_inference.py
        if self.select is not None:
            samples = self.select
        exemplars = self._compute_steps(samples)

        sections = []
        for ex in exemplars:
            sections.append(
                f"Q: {ex['Question']}\n"
                f"A: {ex['cot']}\n"
            )
        context = "\n\n".join(sections)
        return f"{context}\n\nQ: {question}\nLet's think step by step.\nA:"

    def _self_consistency(self, question, samples=11, threshold=0.85, model='all-mpnet-base-v2', return_confidence=False):
        responses = []
        for _ in range(samples):
            prompt = {'role': 'user',
                      'content': question}
            messages, _ = llms.runLLM([prompt])
            responses.append(messages)
        embedder = SentenceTransformer(model)
        embeddings = embedder.encode(responses, show_progress_bar=False)
        matrix = cosine_similarity(embeddings)
        # if sum is 0? no good threshold to keep? :( # -1 takes the lastone
        groups = np.sum(np.tril(matrix) > threshold, 1)
        best_answer = responses[np.argmax(groups)]
        if not return_confidence:
            return best_answer
        else:
            return best_answer, max(groups) / len(responses)

    def apply(self,
              question: str,
              prompt_engineeing_tpye: str = "zero-shot",
              ):
        prompted_q = copy.deepcopy(question)
        if self.beshort:
            prompted_q['content'] = f"Be short and concise.  {prompted_q['content']}"
        prompt_engineeing_tpye = prompt_engineeing_tpye.lower()
        # 1. Zero-shot: Baseline
        if prompt_engineeing_tpye == "zero-shot":
            prompted_q['content'] = self._zero_shot(prompted_q['content'])
        # 2. Language Models are Few‑Shot Learners. https://arxiv.org/abs/2005.14165
        elif prompt_engineeing_tpye == "few-shot":
            prompted_q['content'] = self._few_shot(prompted_q['content'])
        # 3. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. https://arxiv.org/abs/2201.11903
        elif prompt_engineeing_tpye == "few-shot-cot":
            prompted_q['content'] = self._few_shot_cot(prompted_q['content'])
        # 4. Large Language Models are Zero-Shot Reasoners. https://arxiv.org/pdf/2205.11916
        elif prompt_engineeing_tpye == "zero-cot":  # could be improved in "Therefore, blabla Answer: "
            prompted_q['content'] = self._zero_cot(prompted_q['content'])
        # 5. Automatic Chain of Thought Prompting in Large Language Models. https://arxiv.org/abs/2210.03493
        elif prompt_engineeing_tpye == "auto-cot":
            prompted_q['content'] = self._auto_cot(prompted_q['content'])
        # 6. Self-Consistency Improves Chain of Thought Reasoning in Language Models. https://arxiv.org/abs/2203.11171
        elif prompt_engineeing_tpye == "self-consistency":
            prompted_q['content'] = self._self_consistency(
                prompted_q['content'])
        else:
            ValueError("Incorrent prompt engineering selection")
        return prompted_q
