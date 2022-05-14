# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def question_answer(question, text):
    
    #tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #number of tokens in segment A (question)
    num_seg_a = sep_idx+1
    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s for segment embeddings
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
                
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    
    print("\nPredicted answer:\n{}".format(answer.capitalize()))

text = '''FlexiPay 
Features
The FlexiPay - Buy Now Pay Later offers a plethora of features and benefits to meet your every requisite. 
First of Its Kind: No Extra Cost for 15 days.
Zero Charges: Zero Convenience Fees, Zero Processing Fees and No Hidden Charges. 
Anytime Anywhere Credit: Extra cash available 24*7. Enjoy the convenience of availing a credit line from Rs. 1,000 up to Rs. 20,000 instantly. 
Flexible Repayment: FlexiPay comes with pay later options that have convenient repayment tenures. Choose repayment starting from 15 days. 
Flexible Repayment Tenure: Select tenure starting from 15 days up to 90 days at nominal charges per month
Hassle-Free Pay Offs: Pay your utilised principal and interest at the end of your preferred tenure. 
Eligibility
Only pre-approved current account and savings account holding customers of HDFC Bank are eligibile for FlexiPay facility. The loan amount criteria for availing FlexiPay- Pay Later is:
Minimum Loan Amount: Rs. 1,000.
Maximum Loan Amount: Rs. 20,000.
Fees and Charges
Rate of Interest: 
There are four tenures with the respective applicable EMI interest rates on any transaction carried through the FlexiPay- Buy Now Pay Later option
Tenure.
Interest rate
15 days 
No Extra Cost
30 days 
28 interest annually(For eg: Rs 70 per month on purchase of Rs 3000)
60 days
28 interest annually (For eg: Rs 70 per month on purchase of Rs 3000)
90 days
28 interest annually (For eg: Rs 70 per month on purchase of Rs 3000)
Scenario 1: If loan is taken on the 1st for 15 days, the same will be due on 16th, on 16th, customer’s account will be auto debited for the loan amount. 
Note : NO Extra Cost for 15 days has been given as upfront discount to cover for interest charged by the Bank, effectively giving you the benefit of NO Extra Cost. Total amount you repay to the Bank will be equal to the invoice amount of your order.​​​​​​​
Scenario 2 : If the customer has taken tenure of 30 days, with a disbursement on January 10th & due date of Feb 9th . Customer will be charged interest + principal after 30 days from loan booking i.e. on Feb 9th (30 days including date of disbursement) 
Scenario 3 : In case of 60 days tenure, interest will be charged for 30 days on Feb 9th, on 9th march (60 days including date of disbursement) interest for balance 30 days and principal will be recovered.
Note : Similar will be the case for 90 days tenure .
Auto Debit return penal interest: 
A 2 interest rate plus GST at 18 is applicable subject to change as per the Government’s instruction, subject to a minimum of Rs.450 shall be levied. 
Late Payment Fee:
Non-payment or partial repayment of the outstanding amount will attract a late payment penalty of 3 plus GST at 18, subject to change as per the Government’s instruction on the total outstanding due. 
Pre Closure Charges: 
Pre-closure of the FlexiPay service will currently attract a charge of 4 on the balance principal outstanding plus GST at 18, subject to change as per the Government’s instruction. 
Frequently Asked Questions.
​​​​​​​Where can I use FlexiPay? 
FlexiPay - Pay Later is accessible as a payment option at your preferred online platform’s check out page. It caters to your specific needs, in addition to the primary options of Credit Card, Debit Card and EMI. 
To use this option for making a payment, choose ‘FlexiPay’  in the check out page.
How does FlexiPay work? 
With FlexiPay, you are offered a digital credit for up to 90 days. For a tenure of 30, 60 or 90 days, the interest is debited from your account on the due date. The principal amount is recoverable at the end of the chosen tenure.
The most lucrative benefit of this product is the 15-day at NO EXTRA COST, wherein only the principal amount is debited at the end of the selected tenure. 
How can I pay my FlexiPay charges? 
The due amount is automatically debited from your existing HDFC Bank Savings or Current Account.
How does it work? 
There are five simple steps to follow to use this service: 
Select HDFC Bank FlexiPay- Buy Now Pay Later at the check out page on the website. 
Enter your HDFC Bank Registered mobile number.
Select the preferred tenure of your choice, enter the last 4- digits of your HDFC Bank Debit Card. Select the terms and conditions checkbox to proceed further. 
Enter the OTP received on your mobile number to validate the specific transaction. 
You are done. '''
text=text[:512]
question = input("\nPlease enter your question: \n")
while True:
    question_answer(question, text)
    
    flag = True
    flag_N = False
    
    while flag:
        response = input("\nDo you want to ask another question based on this text (Y/N)? ")
        if response[0] == "Y":
            question = input("\nPlease enter your question: \n")
            flag = False
        elif response[0] == "N":
            print("\nBye!")
            flag = False
            flag_N = True
            
    if flag_N == True:
        break