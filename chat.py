import os
from langchain_openai import ChatOpenAI
import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_response(msg):

    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.02)

    context = """ Estimados padres de familia,\n\nEs un placer darles la bienvenida al Liceo Pequeños Querubines, una institución educativa privada dedicada a ofrecer un entorno seguro y estimulante para el desarrollo integral de sus hijos en la primera infancia. Nuestro objetivo es fomentar el liderazgo y el aprendizaje significativo en un ambiente lleno de amor y compromiso. Contamos con dos sedes en Bogotá: Sede A, ubicada en Calle 22 Bis N 93 B 03, y Sede B, en Calle 22 A Bis A N 93 A 14. Para cualquier consulta, pueden contactarnos al teléfono 3058168900 o al correo electrónico liceopequenosquerubines@hotmail.com.\n\nEn nuestro jardín, los niños se dividen en diferentes niveles según su edad y desarrollo: "Mis primeras alitas" para los más pequeños, "Explorers" para niños de tres años, "Discovers" para los de cuatro años, y "Thinkers" para aquellos que están en una etapa de desarrollo afectivo y emocional. Cada nivel está diseñado para satisfacer las necesidades específicas de los niños, promoviendo su independencia y habilidades sociales.\n\nNuestro equipo de profesionales está liderado por la Lic. Luz Deiby Méndez, Directora General, quien ha dedicado su vida a crear un espacio ideal para el crecimiento y aprendizaje de los niños. Junto a ella, la Lic. Marcela Guzmán M, Directora Pedagógica, y la Lic. Leidy Johanna Morales Parra, Directora Administrativa, trabajan incansablemente para asegurar que cada niño reciba una educación de calidad. Además, contamos con el apoyo de nuestra psicóloga, Nohorita Torres S, quien acompaña a los niños y sus familias en su desarrollo emocional.\n\nEn el Liceo Pequeños Querubines, también ofrecemos diversas escuelas formativas para enriquecer la experiencia educativa de los niños. Entre ellas, la Escuela de Karate - Kohai, dirigida por el Sensei John Salguero, la Escuela de Arte "Arlequines de colores" con Miss Juliana y Miss Catalina, la Escuela de Danza Contemporánea "Soul Dance" con Miss Hillary Zapata, y la Escuela de Música con Miss Anny Cortés. Además, ofrecemos clases de natación en convenio con el Club El Castor Modelia y el Club de Científicos liderado por Miss Mariana.\n\nPara asegurar que nuestros estudiantes estén siempre cómodos y presentables, contamos con un uniforme institucional que incluye prendas para el uso diario y deportivo. Las querubinas deben usar una jardinera según el modelo, blusa blanca, pajarita azul oscuro, medias pantalón y zapatos colegiales azules. Los querubines deben llevar pantalón según el modelo, camisa blanca, corbata azul oscura, chaleco azul con el escudo del liceo, medias y zapatos colegiales azules. Los días de educación física, tanto querubinas como querubines deben usar tenis y medias completamente blancas.\n\nFinalmente, les invitamos a ser parte de nuestra familia LPQ completando el formulario de admisión disponible en nuestra página web. Nuestro Departamento de Admisiones se pondrá en contacto con ustedes a la mayor brevedad para brindarles toda la información necesaria. Agradecemos su confianza y esperamos con entusiasmo acompañar a sus hijos en esta maravillosa etapa de sus vidas.\n\nAtentamente,\n\nEl equipo del Liceo Pequeños Querubines
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""Eres un asistente que responde preguntas relacionadas con una institucion educativa.La informacion esta dirigida a
            padres e interesados en tener sus hijos en esta institucion. Las respuestas deben ser educadas e informativas. Usa el siguiente
            contexto para responder las preguntas. Solo responde en español
            Contexto: {context}
            
            """),
            ("user", "{question}")
        ]
    )

    chain = prompt | llm
    
    return chain.invoke(msg).content



if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        while True:
            # sentence = "do you use credit cards?"
            sentence = input("You: ")
            if sentence == "quit":
                break

            resp = get_response(sentence)
            print(resp)

