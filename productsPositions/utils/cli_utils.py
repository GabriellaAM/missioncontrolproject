"""
Utilitários para interface de linha de comando (CLI)
Funções compartilhadas para validação de input e formatação
"""

def obter_input(mensagem, tipo=str, opcoes=None, obrigatorio=True):
    """Função auxiliar para obter input do usuário com validação"""
    while True:
        valor = input(mensagem).strip()
        
        if not valor and obrigatorio:
            print("⚠️  Este campo é obrigatório!")
            continue
        
        if not valor and not obrigatorio:
            return None
        
        # Validação de tipo
        if tipo == float or tipo == int:
            try:
                valor_convertido = tipo(valor)
                # Só validar > 0 se for obrigatório
                if obrigatorio:
                    if tipo == float and valor_convertido <= 0:
                        print("⚠️  O valor deve ser maior que zero!")
                        continue
                    if tipo == int and valor_convertido <= 0:
                        print("⚠️  O valor deve ser maior que zero!")
                        continue
                return valor_convertido
            except ValueError:
                print(f"⚠️  Por favor, insira um número válido!")
                continue
        
        # Validação de opções
        if opcoes and valor not in opcoes:
            print(f"⚠️  Opções válidas: {', '.join(opcoes)}")
            continue
        
        return valor


def validar_data(data_str, nome_campo="Data"):
    """Valida formato de data YYYY-MM-DD"""
    from datetime import datetime
    
    while True:
        if len(data_str) == 10 and data_str[4] == '-' and data_str[7] == '-':
            try:
                # Tentar parsear para garantir que é uma data válida
                datetime.strptime(data_str, "%Y-%m-%d")
                return data_str
            except ValueError:
                print(f"⚠️  Data inválida! Use YYYY-MM-DD")
                data_str = input(f"{nome_campo} (YYYY-MM-DD): ").strip()
        else:
            print("⚠️  Formato inválido! Use YYYY-MM-DD")
            data_str = input(f"{nome_campo} (YYYY-MM-DD): ").strip()


def obter_data(mensagem, obrigatorio=True):
    """Obtém uma data do usuário com validação"""
    data = input(mensagem).strip()
    if not data and not obrigatorio:
        return None
    return validar_data(data, mensagem.replace(":", ""))


def imprimir_titulo(titulo, largura=60):
    """Imprime um título formatado"""
    print("=" * largura)
    print(f"  {titulo}")
    print("=" * largura)
    print()


def imprimir_secao(titulo, largura=60):
    """Imprime uma seção formatada"""
    print("\n" + "=" * largura)
    print(f"  {titulo}")
    print("=" * largura)
    print()

