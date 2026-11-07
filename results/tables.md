Model: qwen2.5:3b   n = 100   k = 5   Recall@5 = 33.0%

| Condition | Correct % | Incorrect % | Abstain % | Selective acc. % | Calls/q | Tokens/q | Retrieval ms/q |
|---|---|---|---|---|---|---|---|
| A: Direct | 14.0 | 84.0 | 2.0 | 14.3 | 1 | 67 | 0.0 |
| B: + Abstain | 9.0 | 14.0 | 77.0 | 39.1 | 1 | 82 | 0.0 |
| C: + Retrieval | 18.0 | 28.0 | 54.0 | 39.1 | 1 | 838 | 35.0 |
| D: + Verify | 16.0 | 27.0 | 57.0 | 37.2 | 2 | 1818 | 35.0 |

Checker decisions (rows = C's draft):

| C draft | Keep | Corrected-right | Corrected-wrong | Withdraw |
|---|---|---|---|---|
| correct | 10 | 6 | 2 | 0 |
| incorrect | 10 | 0 | 13 | 5 |
| abstain | 1 | 0 | 2 | 51 |

Paired differences, percentage points (95% bootstrap CI):
  A->B: correct -5.0 [-10.0, +0.0]  incorrect -70.0 [-78.0, -61.0]
  B->C: correct +9.0 [+1.0, +17.0]  incorrect +14.0 [+4.0, +24.0]
  C->D: correct -2.0 [-5.0, +0.0]  incorrect -1.0 [-7.0, +5.0]

B abstained on 6 questions that A answered correctly
C wrong answers with gold missing from top-k: 22 / 28
D withdrew a correct C answer while gold was missing from passages: 0
Checker outputs that could not be parsed: 0
