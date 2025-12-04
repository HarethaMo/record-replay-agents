"""Toy GSM8K-style datasets for quick experimentation.

NOTE:
    For a *real* experiment, you should replace these with actual samples
    from GSM8K and GSM8K-Hard. Here we provide a handful of synthetic
    problems in the same style so that the project runs out-of-the-box.

Each sample is a dict:
    {
        "id": str,
        "question": str,
        "answer": str  # numeric answer as a string
    }
"""

from __future__ import annotations

from typing import List, Dict

BASIC_MATH_SAMPLES: List[Dict[str, str]] = [
    {
        "id": "basic_math_1",
        "question": "5 + 15 + 10 = ?",
        "answer": "30",
    },
    {
        "id": "basic_math_2",
        "question": "10 + 5 + 7 + 4 = ?",
        "answer": "26",
    },
    {
        "id": "basic_math_3",
        "question": "10 + 8 + 20 + 7 + 5 = ?",
        "answer": "50",
    },
    {
        "id": "basic_math_4",
        "question": "(1 + 9) - (2 + 3) = ?",
        "answer": "5",
    },
    {
        "id": "basic_math_5",
        "question": "20 - 5 + 10 - 3 = ?",
        "answer": "22",
    },
    
]

GSM8K_SAMPLES: List[Dict[str, str]] = [
    # {
    #     "id": "gsm8k_1",
    #     "question": "Alice has 3 packs of stickers. Each pack has 8 stickers. She gives 5 stickers to her friend. How many stickers does she have left?",
    #     "answer": "19",
    # },
    # {
    #     "id": "gsm8k_2",
    #     "question": "A classroom has 6 rows of desks, and each row has 4 desks. Each desk seats 2 students. How many students can sit in the classroom?",
    #     "answer": "48",
    # },
    {
        "id": "gsm8k_3",
        "question": "Tom runs 4 kilometers every day from Monday to Friday. On Saturday he runs 6 kilometers and rests on Sunday. How many kilometers does he run in a week?",
        "answer": "26",
    },
    {
        "id": "gsm8k_4",
        "question": "There are 5 boxes of apples. Each box contains 12 apples. If 17 apples are eaten, how many apples remain?",
        "answer": "43",
    },
    {
        "id": "gsm8k_5",
        "question": "A book has 250 pages. Sara reads 30 pages each day for 5 days and then 10 pages on the sixth day. How many pages does she still need to read?",
        "answer": "90",
    },
]


GSM8K_HARD_SAMPLES: List[Dict[str, str]] = [
    {
        "id": "gsm8k_hard_1",
        "question": "A factory produces 125 widgets per hour. It runs 7 hours a day for 5 days. 15% of the widgets are defective and thrown away. How many good widgets remain?",
        "answer": "3725",
    },
    {
        "id": "gsm8k_hard_2",
        "question": "A shop buys 80 notebooks at $2.50 each and 40 pens at $1.20 each. It sells all items with a 25% profit. What is the total revenue from selling all notebooks and pens?",
        "answer": "304",
    },
    {
        "id": "gsm8k_hard_3",
        "question": "A tank is 3/5 full of water. It can hold 900 liters when full. 120 liters are pumped out and then 90 liters are added back. How many liters of water are in the tank now?",
        "answer": "510",
    },
    # {
    #     "id": "gsm8k_hard_4",
    #     "question": "A bus trip costs $2.40 per ride. Emma buys a monthly pass for $60 that allows her to ride 30 times for free and then pay only $1.00 per ride afterward. If she rides the bus 45 times this month, how much money does she spend in total including the pass?",
    #     "answer": "75",
    # },
    # {
    #     "id": "gsm8k_hard_5",
    #     "question": "A farmer sells eggs in cartons of 12. On Monday he sells 18 cartons, on Tuesday 15 cartons, and on Wednesday 21 cartons. 8% of all eggs sold are cracked and must be discarded. How many good eggs does he sell over these three days?",
    #     "answer": "600",
    # },
]
