from pydantic_ai import Agent, RunContext

system_prompt = """
A sua funcão é dar contexto ao texto que você vai receber, baseado
no diretório que está na função `directory_context`. Cada fragmento de
texto que você irá receber é um dos casos abaixo:
- A junção de varios arquivos dentro de uma pasta raiz, no qual estará
marcado durante o texto a transição entre arquivos.
- Ou algum arquivo que foi separado por ser muito grande. Neste caso,
no inicio do arquivo estara indicando a qual part se refere:
Parte (m/n), onde m <= n.
Você deve retornar em menos de 200 caracters o contexto em que este
documento está, assim como os principais tópicos e tecnologias que
são abordados. O objetivo é dar contexto a quais arquivos estão em
cada fragmento e quais os subdiretorios que tem,
bem como o que é abordado.
"""

contextual_agent = Agent(
    'openai:gpt-4o',
    system_prompt=system_prompt,
    result_type=str,
    deps_type=str,
    retries=2,
)


@contextual_agent.system_prompt
def root_folder(ctx: RunContext[str]) -> str:
    return f'O texto pertence a pasta {ctx.deps}, situada na raiz'
