import time
from typing import AsyncIterator

from langchain_core.prompts import ChatPromptTemplate

from app.logger import get_logger
from app.services.llm import routed_chat_llm

logger = get_logger("wespeak.feedback")

SYSTEM_PROMPT = """You are an English reading comprehension coach for language learners.
You will receive:
1. A book passage
2. A student's spoken response about that passage

Your job depends on what the student says:
- If the student says they do not understand or asks for a summary, provide a clear and friendly summary of the passage in 3-5 sentences
- If the student provides their own summary, evaluate whether they grasped the overall context and flow of the passage
  - Focus on the big picture: did they capture the main theme and how the key ideas connect?
  - Do NOT nitpick minor details
  - Do NOT quote sentences from the passage verbatim in your feedback
- Give your response in 3-4 sentences
- Be encouraging and specific
- Respond in English only"""

BOOK_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("human", """Book passage:
The Amazon rainforest, often called the 'lungs of the Earth', produces about 20% of the world's oxygen. It covers over 5.5 million square kilometers across nine countries, with Brazil containing the largest portion. Deforestation due to agriculture and logging has destroyed nearly 20% of the original forest over the past 50 years, threatening thousands of species.

Student's summary:
The Amazon is a very big forest. It makes oxygen. Many animals live there but some people cut trees so it is dangerous for animals."""),
    ("ai", "You understood that the Amazon is vital for oxygen production and that deforestation is putting it at risk — that cause-and-effect is exactly the key point of this passage."),
    ("human", """Book passage:
The Industrial Revolution, which began in Britain in the late 18th century, transformed societies from agrarian economies to manufacturing-based ones. Steam-powered machines replaced manual labor, enabling mass production in factories. While this created economic growth and urbanization, it also led to poor working conditions, child labor, and significant environmental pollution.

Student's summary:
I think it was about some machines and people working. There were factories. I'm not sure what the main point was."""),
    ("ai", "You picked up on the machines and factories, but the key point of this passage is that steam-powered machines fundamentally transformed society — people shifted from farming to factory labor, which brought economic growth but also serious problems like poverty and pollution."),
    ("human", """Book passage:
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?" So she was considering in her own mind whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

Student's summary:
I can't understand this. Can you give me a summary?"""),
    ("ai", "Of course! In this passage, a girl named Alice is sitting by a riverbank with her sister, feeling bored and restless because she has nothing to do. She glances at her sister's book but finds it dull since it has no pictures. Just as she is lazily thinking about making a daisy chain, a White Rabbit suddenly runs past her — sparking her curiosity and setting her adventure in motion."),
    ("human", """Book passage:
Bees are among the most important creatures on Earth. They pollinate about one-third of the food we eat, transferring pollen between flowers as they collect nectar. Without bees, many plants would be unable to reproduce, leading to a collapse of ecosystems that depend on those plants. Alarmingly, bee populations worldwide have been declining due to pesticide use, habitat loss, and disease.

Student's summary:
Bees are really important because they help plants grow by moving pollen around. If bees disappear, many ecosystems and food sources could be in danger. The problem is that bee numbers are going down because of things like chemicals and losing their homes."""),
    ("ai", "You understood both why bees matter and the threat they face, and connected those two ideas well — that's exactly the key point of this passage."),
    ("human", """Book passage:
On July 20, 1969, NASA's Apollo 11 mission successfully landed the first humans on the Moon. Astronauts Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above. The achievement was the culmination of years of scientific effort and political competition during the Cold War Space Race between the United States and the Soviet Union.

Student's summary:
Some astronauts went to the Moon in 1969. It was a big event and people were excited."""),
    ("ai", "You got the basic event, but the key point of this passage is that the Moon landing was the culmination of years of political and scientific competition between the US and the Soviet Union — that rivalry is what gives the achievement its real historical significance."),
    ("human", """Book passage:
Social media platforms have transformed the way people communicate, share information, and form communities. While they allow individuals to connect across the globe instantly, they also contribute to the spread of misinformation and can negatively affect mental health, particularly among teenagers. Researchers have found that excessive use is linked to anxiety, depression, and a distorted sense of self-worth.

Student's summary:
Social media connects people around the world, but it also has dark sides like spreading false information and making people, especially teenagers, feel bad about themselves."""),
    ("ai", "You understood that social media is a double-edged sword — powerful for connection but harmful when overused — and that's exactly the key point of this passage."),
    ("human", """Book passage:
Romeo and Juliet is a tragedy by William Shakespeare about two young lovers from rival families in Verona. Their secret romance defies the long-standing feud between the Montagues and the Capulets. A series of misunderstandings and impulsive decisions ultimately leads to both of their deaths, leaving their families to grieve and reconcile.

Student's summary:
Romeo and Juliet is a love story. They loved each other very much and both died at the end."""),
    ("ai", "You got that it's a love story, but you missed that what makes it a tragedy is the families' bitter feud combined with a chain of misunderstandings — those are what turn the romance into disaster."),
    ("human", """Book passage:
The law of supply and demand is one of the most fundamental principles in economics. When demand for a product rises while supply remains the same, prices tend to go up. Conversely, when supply exceeds demand, prices fall. This relationship constantly shapes market decisions made by both consumers and businesses.

Student's summary:
Supply and demand explains how prices change in markets. When more people want something but there isn't enough of it, prices go up. When there's too much of something nobody wants, prices go down."""),
    ("ai", "You clearly understood that price is driven by the balance between demand and supply — that's exactly the key point of this passage."),
    ("human", """Book passage:
Climate change refers to long-term shifts in global temperatures and weather patterns. While some natural factors contribute, human activities — particularly the burning of fossil fuels — have been the primary driver since the industrial era. Rising temperatures lead to melting ice caps, rising sea levels, and more frequent extreme weather events, threatening communities worldwide.

Student's summary:
The weather is changing and it's getting hotter. Ice is melting and there are more storms."""),
    ("ai", "You described the effects, but you missed that the key point of this passage is the cause — it's primarily human activity, specifically burning fossil fuels, that is driving these changes."),
    ("human", """Book passage:
Nelson Mandela spent 27 years in prison for his opposition to South Africa's apartheid system, a policy of racial segregation enforced by the government. After his release in 1990, rather than seeking revenge, he worked to build a peaceful, democratic South Africa. In 1994, he became the country's first Black president, a symbol of forgiveness and reconciliation.

Student's summary:
Nelson Mandela was imprisoned for fighting against apartheid, but after being released he chose peace over revenge and became the first Black president of South Africa."""),
    ("ai", "You understood the full arc — imprisonment, release, and his choice of forgiveness over revenge — which is exactly the key point of this passage."),
    ("human", """Book passage:
The Japanese concept of 'ikigai' refers to a person's reason for being — the intersection of what you love, what you are good at, what the world needs, and what you can be paid for. It is often cited as one of the reasons for the longevity and happiness of people in Okinawa, Japan. Finding one's ikigai is believed to give life a sense of purpose and direction.

Student's summary:
Ikigai is a Japanese word. It's about being happy and living long."""),
    ("ai", "You got that ikigai relates to happiness and longevity, but the key point of this passage is what ikigai actually means — finding your purpose where your passion, your skills, and your contribution to the world all meet. That sense of purpose is what drives everything else."),
    ("human", """Book passage:
Artificial intelligence is increasingly being used in healthcare to assist doctors in diagnosing diseases. AI systems can analyze medical images, patient records, and research data far faster than humans, often identifying patterns that doctors might miss. While AI shows great promise, experts warn that it should assist rather than replace human judgment, as ethical and contextual decisions still require human oversight.

Student's summary:
AI is being used in medicine to help doctors find diseases faster. It can analyze data quickly and spot things humans might miss, but doctors should still be in charge because some decisions need human thinking."""),
    ("ai", "You understood both what AI can do and why human oversight is still essential — that balance between capability and responsibility is exactly the key point of this passage."),
    ("human", """Book passage:
The trolley problem is a classic ethical dilemma used in philosophy and psychology. It asks: if a runaway trolley is heading toward five people tied to the tracks, would you pull a lever to divert it to another track where only one person is tied? The dilemma explores the tension between utilitarian thinking — saving the most lives — and the moral discomfort of actively causing harm to an innocent person.

Student's summary:
The trolley problem is about a trolley that can kill people. You have to decide what to do."""),
    ("ai", "You got the basic scenario, but you missed that the key point of this passage is the ethical tension — the conflict between saving the most lives and the moral discomfort of directly causing harm to one innocent person."),
    ("human", """Book passage:
A black hole is a region of space where gravity is so strong that nothing — not even light — can escape from it. They form when massive stars collapse under their own gravity at the end of their lives. The boundary around a black hole, beyond which nothing can return, is called the event horizon. Despite being invisible, black holes can be detected by observing their effects on nearby matter and light.

Student's summary:
I really can't understand this at all. What is it about?"""),
    ("ai", "No problem — let me break it down! A black hole is an extremely powerful region in space where gravity is so intense that even light gets pulled in and cannot escape. They are born when giant stars die and collapse in on themselves. Even though we cannot see black holes directly, scientists can tell they are there by watching how they affect the space and matter around them."),
    ("human", """Book passage:
The French Revolution, which began in 1789, was a period of radical political and social transformation in France. Fueled by widespread poverty, inequality, and resentment toward the monarchy, the French people revolted against King Louis XVI. The revolution led to the abolition of the monarchy, the rise of new democratic ideals, and a period of violent political upheaval known as the Reign of Terror.

Student's summary:
I don't get this passage at all. Can you explain it to me?"""),
    ("ai", "Sure! The French Revolution was a massive uprising in France in the late 1700s, where ordinary people who were fed up with poverty and an unfair system rose up against their king. The old monarchy was overthrown, and France began moving toward a system based on equality and democracy. However, the transition was not peaceful — it came with a period of extreme violence and political chaos before things could stabilize."),
    ("human", """Book passage:
F. Scott Fitzgerald's The Great Gatsby is set in the 1920s and follows Jay Gatsby, a wealthy and mysterious man who throws lavish parties in hopes of reuniting with his lost love, Daisy Buchanan. Gatsby's obsession with Daisy represents his longing to recapture the past and achieve the American Dream. The novel ultimately reveals the hollowness of wealth and the impossibility of reversing time.

Student's summary:
I can't understand this text. What does it mean?"""),
    ("ai", "This passage is about a novel called The Great Gatsby, set in the glamorous 1920s. The main character, Gatsby, is a rich man who throws enormous parties with one secret goal: to win back a woman named Daisy whom he loved long ago. The story is really about how Gatsby's dream of recreating the past is both beautiful and doomed — and how chasing wealth and something already gone can lead to emptiness."),
    ("human", """Book passage:
DNA, or deoxyribonucleic acid, is the molecule that carries the genetic instructions for the development, functioning, and reproduction of all known living organisms. It is structured as a double helix — two strands twisted around each other — and is made up of four chemical bases. The specific sequence of these bases determines the genetic information carried by a segment of DNA, known as a gene.

Student's summary:
This is too hard for me. I don't understand anything. Can you summarize it?"""),
    ("ai", "Of course! DNA is basically the instruction manual found inside every living thing — it tells your body how to grow, function, and reproduce. It has a famous twisted ladder shape, built from four chemical building blocks arranged in a specific order. That specific order is what makes up your genes, which determine everything from your eye color to how your cells work."),
    ("human", """Book passage:
Inflation refers to the general increase in prices of goods and services over time, which reduces the purchasing power of money. It is typically measured by tracking the cost of a standard set of goods, known as a consumer price index. Moderate inflation is considered normal in a healthy economy, but high inflation can erode savings, increase the cost of living, and destabilize financial systems.

Student's summary:
I'm not sure what this is saying. Can you explain?"""),
    ("ai", "Sure! Inflation simply means that over time, things cost more money — so the same amount of money buys you less than it used to. Economists track this by watching how the prices of everyday items change over time. A little bit of inflation is actually normal and healthy for an economy, but when prices rise too fast, it can be tough on people's savings and daily lives."),
    ("human", """Book passage:
In Plato's Allegory of the Cave, prisoners are chained in a cave and can only see shadows projected on a wall, which they mistake for reality. When one prisoner escapes and sees the real world and the sun, he realizes the shadows were mere illusions. Plato uses this allegory to suggest that most people mistake limited perceptions for the whole truth, and that genuine knowledge requires stepping beyond what is immediately visible.

Student's summary:
I have no idea what this is about. Please give me a summary."""),
    ("ai", "This is a famous philosophical story by Plato about knowledge and truth. Imagine people who have been chained in a dark cave their whole lives, only ever seeing shadows on a wall — they believe those shadows are the real world. When one person finally escapes and sees actual sunlight, it is overwhelming but eye-opening. Plato's point is that most of us, like those prisoners, only see a limited version of reality, and that true understanding requires the effort to look beyond what is right in front of us."),
    ("human", """Book passage:
Ocean acidification is the ongoing decrease in the pH of Earth's oceans, primarily caused by the absorption of carbon dioxide from the atmosphere. As CO2 dissolves in seawater, it forms carbonic acid, making the ocean more acidic. This increased acidity threatens marine life, particularly organisms like corals and shellfish that rely on calcium carbonate to build their shells and skeletons, and could destabilize entire ocean ecosystems.

Student's summary:
I couldn't understand this. What's it about?"""),
    ("ai", "No worries! This passage is about how our oceans are gradually becoming more acidic, mainly because they are absorbing too much carbon dioxide from the air. When that gas mixes with seawater, it creates an acid that slowly changes the ocean's chemistry. This is a serious problem because many sea creatures — like corals and shellfish — struggle to build their shells in more acidic water, and if it continues, it could damage entire ocean ecosystems."),
    ("human", """Book passage:
A blockchain is a type of digital ledger that records transactions across a network of computers in a way that makes them nearly impossible to alter. Each transaction is grouped into a block, and these blocks are linked together in a chronological chain. Because no single authority controls the blockchain, it is considered decentralized and transparent. This technology forms the foundation of cryptocurrencies like Bitcoin.

Student's summary:
I really don't understand this text at all. Can you give me a simpler explanation?"""),
    ("ai", "Of course! A blockchain is basically a shared digital record book that keeps track of transactions, but instead of being stored in one place, copies of it are spread across many computers at once. Every new transaction gets added as a new entry, and once recorded it is nearly impossible to change or fake. Because no single person or organization is in charge of it, people trust it — and this is the technology that makes cryptocurrencies like Bitcoin work."),
    ("human", """Book passage:
Confirmation bias is the tendency for people to search for, interpret, and remember information in a way that confirms their pre-existing beliefs. This cognitive bias leads individuals to give more weight to evidence that supports what they already think and to dismiss information that challenges it. Confirmation bias can reinforce false beliefs, fuel political polarization, and make it difficult for people to change their minds even when faced with strong contradictory evidence.

Student's summary:
I don't understand what this is saying. Please summarize it for me."""),
    ("ai", "Sure! Confirmation bias is a mental habit where people naturally pay more attention to information that agrees with what they already believe, and tend to ignore anything that contradicts them. For example, if you strongly believe something, you will notice and remember things that prove you right — and brush off things that prove you wrong. This can be a real problem because it makes it very hard to change our minds, even when the evidence clearly points in another direction."),
    ("human", """Book passage:
The Silk Road was an ancient network of trade routes that connected China with the Mediterranean world, spanning over 6,000 kilometers. Active from around 130 BCE to the 1450s CE, it facilitated the exchange not just of goods like silk, spices, and precious metals, but also of ideas, religions, technologies, and diseases. The Silk Road played a key role in shaping the cultures and economies of civilizations along its path.

Student's summary:
I can't understand this passage. Can you explain what it's about?"""),
    ("ai", "Of course! The Silk Road was a vast network of ancient trade routes stretching thousands of kilometers, connecting China to Europe and the Middle East. For over a thousand years, merchants used these routes to trade goods like silk and spices, but people along the way also exchanged ideas, religions, inventions, and even diseases. It was essentially one of the world's earliest forms of globalization, deeply shaping the cultures and histories of countless civilizations along the way."),
    ("human", """Book passage:
{book_content}

Student's summary:
{user_summary}

Please provide feedback on the student's summary.""")
])


# 프론트가 이 텍스트를 그대로 TTS로 읽어주므로(expo-speech) 영어로 고정한다.
# SYSTEM_PROMPT의 "3-4 sentences / encouraging / English only" 가이드와 톤을 맞췄다.
EMPTY_INPUT_MESSAGE = (
    "I couldn't hear anything in your recording. "
    "Could you try again and tell me what you understood from the passage? "
    "Even one or two sentences is a great start."
)


def _extract_contents(raw_messages: list[dict]) -> dict:
    """
    백엔드가 보내는 메시지 형식:
    [
      {"role": "user",      "content": "<book passage>"},
      {"role": "assistant", "content": "Give me a summary."},
      {"role": "user",      "content": "<user's summary>"}
    ]
    """
    if len(raw_messages) < 3:
        book_content = raw_messages[0]["content"] if raw_messages else ""
        user_summary = raw_messages[-1]["content"] if raw_messages else ""
    else:
        book_content = raw_messages[0]["content"]
        user_summary = raw_messages[2]["content"]

    return {"book_content": book_content,
            "user_summary": user_summary}


async def get_feedback_stream(messages: list[dict]) -> AsyncIterator[str]:
    """토큰 단위 스트리밍. gRPC FeedbackService(spec 004)에서 사용."""
    logger.info("feedback request (stream)")
    start = time.perf_counter()
    user_input = _extract_contents(messages)

    # 사용자가 아무 말도 하지 않으면 stt_service.transcribe()가 빈 문자열을 반환한다
    # (vad_filter=True가 무음 구간을 전부 제거 → 세그먼트 0개). 그대로 프롬프트에 넣으면
    # "Student's summary:" 뒤가 비어버리는데, BOOK_PROMPT_TEMPLATE의 few-shot 23개는 전부
    # 학생이 실제 답변을 제출한 경우뿐이라(빈 답변 예시 0개, 되묻는 예시 0개) 모델이
    # "요약을 제출했다"고 전제하고 칭찬해버린다. LLM에 넘기기 전에 차단한다.
    # LLM 동시성 슬롯을 잡기 전에 검사해야 슬롯도 낭비하지 않는다.
    if not user_input["user_summary"].strip():
        logger.info("feedback skipped - empty transcript")
        yield EMPTY_INPUT_MESSAGE
        return

    try:
        async with routed_chat_llm() as llm:
            bookChain = BOOK_PROMPT_TEMPLATE | llm
            async for chunk in bookChain.astream(user_input):
                if chunk.content:
                    yield chunk.content
        logger.info("feedback completed (stream) - %.1fms", (time.perf_counter() - start) * 1000)
    except Exception as e:
        logger.error("feedback failed (stream) - %.1fms error=%s", (time.perf_counter() - start) * 1000, e, exc_info=True)
        raise


async def get_feedback(messages: list[dict]) -> str:
    """기존 REST(/feedback)용 — get_feedback_stream()의 토큰을 모아 완성된 문자열로 반환한다.
    REST 응답 동작은 이전(ainvoke 기반)과 동일하게 유지된다."""
    return "".join([token async for token in get_feedback_stream(messages)])
