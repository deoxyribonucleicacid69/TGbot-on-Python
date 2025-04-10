def proverka_str(p):
    import re
    p1=re.split("[,.;: ]",p)

    p1=''.join(p1)
    if p1.isalpha():
        return True
    return False
def proverka_chisla(p):
    import re
    p1 = re.split("[,.;: ]", p)
    p1 = ''.join(p1)
    if p1.isdigit():
        return True
    return False
