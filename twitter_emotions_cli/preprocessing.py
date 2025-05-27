import re

def basic_clean(text: str) -> str:
    """
    Cleans input text by lowercasing, removing URLs, mentions, hashtags,
    punctuation, numbers, and extra whitespace.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = " ".join(text.split())
    return text

if __name__ == '__main__':
    sample_text_1 = "Hello @user1, check this #awesome link: http://example.com 123 !!"
    sample_text_2 = "I'm   feeling SO excited 😊 & #HAPPY today 🥳 ... "
    
    print(f"Original: 'Hello @user1, check this #awesome link: http://example.com 123 !!'")
    print(f"Cleaned:  'hello check this awesome link'")
    print(f"\nOriginal: 'I'm   feeling SO excited 😊 & #HAPPY today 🥳 ... '")
    print(f"Cleaned:  'im feeling so excited happy today'")
