import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from aether.data import get_batch
from aether.model import BigramModel
from aether.tokenizer import CharTokenizer


def main():
    text = (
        """A good security assessment starts long before the first scan runs. The team sits down with the client, maps out what actually matters, and agrees on what is in scope and what is not. Servers, laptops, cloud accounts and the people who use them all tell a story, and the job is to read that story honestly. Most of the time the interesting problems are not exotic. They are an old password policy, a forgotten test server, or a backup bucket that someone left open on a Friday afternoon.
 
        Networks are a lot like cities. There are main roads, back alleys, and doors that nobody remembers installing. A tester walks the streets the way an attacker would, but leaves a clear trail and writes everything down. The report that comes out of it should be useful to an engineer at two in the morning: what was found, why it matters, and how to fix it without breaking production.
        
        Writing code for this kind of work rewards patience. Small tools that do one thing well are easier to trust than a giant framework nobody has read. A script that parses logs, a helper that checks certificate dates, a tiny model that learns the rhythm of a language: each one teaches something about how systems behave when nobody is watching. Keep the functions short, name things clearly, and test with real data, not only with the examples you wrote yourself.
        
        Language models, at their simplest, are just counting machines. Look at a few characters, remember what came next, and tally it up. Given enough text, the counts start to look like habits. After the letters t and h, the letter e is very likely. After q, almost always u. Nothing in the table understands meaning, yet the output can feel strangely familiar, like a sentence heard through a wall. More context makes the guesses sharper, but also makes the model memorize instead of generalize, so the trick is picking an order that is long enough to sound natural and short enough to still surprise you.
        
        Good habits compound. Write things down, automate the boring parts, review your own work as if a stranger wrote it, and never stop being curious about how things break. The best engineers are not the ones who never make mistakes. They are the ones who leave the system a little clearer than they found it, and who share what they learned so the next person starts one step ahead."""
    )
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    model = BigramModel(tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(model, model.loss)

    for step in range(500):
        inputs, targets = get_batch(token_ids, batch_size=32, context_length=8)
        loss, grads = loss_and_grad(inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 50 == 0:
            print(f"step {step}: loss {loss.item():.3f}")

    start = mx.array([[0]])
    generated = model.generate(start, max_new_tokens=200)
    print(tokenizer.decode(generated[0].tolist()))

if __name__ == "__main__":
    main()