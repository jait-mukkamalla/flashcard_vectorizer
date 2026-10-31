"""Write a small demo .apkg (96 real cards, 6 decks, 12 topics) for trying the app without Anki.

    python benchmarks/make_demo_deck.py demo.apkg
"""
import sys

from flashcard_vectorizer.synthetic import build_apkg

TOPICS = {
    ("Biology", "cell biology"): [
        ("What organelle produces most of the cell's ATP?", "The mitochondrion, through cellular respiration."),
        ("What does the cell membrane do?", "It controls which substances enter and leave the cell."),
        ("What is the function of the ribosome?", "It assembles proteins from amino acids using mRNA."),
        ("Which organelle packages and ships proteins?", "The Golgi apparatus."),
        ("What is the nucleus responsible for?", "Storing DNA and controlling gene expression."),
        ("What do lysosomes contain?", "Digestive enzymes that break down waste."),
        ("What is osmosis?", "Diffusion of water across a semipermeable membrane."),
        ("What is the role of chloroplasts?", "They carry out photosynthesis in plant cells."),
    ],
    ("Biology", "genetics"): [
        ("What is a gene?", "A segment of DNA that codes for a protein or RNA."),
        ("What is an allele?", "A variant form of a gene."),
        ("What is the difference between genotype and phenotype?", "Genotype is the genetic makeup, phenotype the observable traits."),
        ("What did Mendel's law of segregation state?", "Allele pairs separate during gamete formation."),
        ("What is a dominant allele?", "One that masks the effect of a recessive allele."),
        ("What is DNA replication?", "Copying DNA before cell division using DNA polymerase."),
        ("What is a point mutation?", "A change in a single nucleotide of the DNA sequence."),
        ("What does transcription produce?", "An mRNA copy of a gene, made by RNA polymerase."),
    ],
    ("Chemistry", "acids and bases"): [
        ("What is the pH of a neutral solution?", "7 at 25 degrees Celsius."),
        ("Define a Bronsted-Lowry acid.", "A proton donor."),
        ("What is a buffer solution?", "A solution that resists changes in pH when acid or base is added."),
        ("What does a strong acid do in water?", "It dissociates completely, releasing hydrogen ions."),
        ("What is the conjugate base of HCl?", "The chloride ion."),
        ("What is titration used for?", "Finding the concentration of an acid or base by neutralisation."),
        ("What ion do bases release in water?", "Hydroxide ions."),
        ("What colour does litmus turn in acid?", "Red."),
    ],
    ("Chemistry", "organic chemistry"): [
        ("What is a functional group?", "A group of atoms that gives a molecule its characteristic reactions."),
        ("What is an alkane?", "A hydrocarbon with only single carbon-carbon bonds."),
        ("What defines an alcohol?", "A hydroxyl group bonded to a saturated carbon."),
        ("What is an isomer?", "A compound with the same molecular formula but a different structure."),
        ("What is esterification?", "A carboxylic acid reacting with an alcohol to form an ester and water."),
        ("What does an alkene contain?", "At least one carbon-carbon double bond."),
        ("What is polymerisation?", "Joining many monomers into a long chain molecule."),
        ("What is a benzene ring?", "A stable six-carbon ring with delocalised electrons."),
    ],
    ("Physics", "thermodynamics"): [
        ("State the first law of thermodynamics.", "Energy is conserved: change in internal energy equals heat added minus work done."),
        ("What does entropy measure?", "The disorder of a system, which tends to increase."),
        ("What is specific heat capacity?", "The energy needed to raise one kilogram of a substance by one kelvin."),
        ("What is absolute zero?", "0 kelvin, the lowest possible temperature."),
        ("How does heat flow between two bodies?", "From the hotter body to the colder one."),
        ("What is an adiabatic process?", "One in which no heat is exchanged with the surroundings."),
        ("What is conduction?", "Heat transfer through direct contact between particles."),
        ("What does the second law of thermodynamics say?", "The entropy of an isolated system never decreases."),
    ],
    ("Physics", "electromagnetism"): [
        ("State Ohm's law.", "Voltage equals current times resistance."),
        ("What is a magnetic field?", "A region where a moving charge or magnet feels a force."),
        ("What does Faraday's law describe?", "A changing magnetic flux induces an electromotive force."),
        ("What is capacitance?", "The ability of a component to store electric charge."),
        ("What is the unit of electric current?", "The ampere."),
        ("What is electromagnetic induction?", "Generating voltage by changing the magnetic field through a circuit."),
        ("What is Coulomb's law?", "The force between two charges is proportional to the product of charges over distance squared."),
        ("In a series circuit, how does current behave?", "It is the same through every component."),
    ],
    ("History", "Roman history"): [
        ("Who was the first Roman emperor?", "Augustus, formerly Octavian."),
        ("What event began in 44 BC?", "Julius Caesar was assassinated by senators."),
        ("What were the Punic Wars?", "Three wars between Rome and Carthage."),
        ("What was the Roman Senate?", "The governing council of the Roman Republic."),
        ("Who crossed the Rubicon in 49 BC?", "Julius Caesar, starting a civil war."),
        ("What happened to the Western Roman Empire in 476 AD?", "It fell when Odoacer deposed Romulus Augustulus."),
        ("What was the Colosseum used for?", "Gladiatorial contests and public spectacles."),
        ("Who was Hannibal?", "The Carthaginian general who crossed the Alps to attack Rome."),
    ],
    ("History", "World War II"): [
        ("When did World War II begin in Europe?", "1939, when Germany invaded Poland."),
        ("What was D-Day?", "The Allied invasion of Normandy on 6 June 1944."),
        ("What was the Holocaust?", "The Nazi genocide of six million Jews."),
        ("What happened at Pearl Harbor?", "Japan attacked the US Pacific fleet in December 1941."),
        ("What was the Battle of Stalingrad?", "A decisive Soviet victory over Germany on the Eastern Front."),
        ("What were the Axis powers?", "Germany, Italy and Japan."),
        ("How did the war in the Pacific end?", "After atomic bombs fell on Hiroshima and Nagasaki."),
        ("What was the Blitz?", "German bombing of British cities in 1940 and 1941."),
    ],
    ("Computer Science", "programming"): [
        ("What is a variable?", "A named location in memory that stores a value."),
        ("What does a for loop do?", "Repeats a block of code a set number of times."),
        ("What is recursion?", "A function that calls itself to solve smaller subproblems."),
        ("What is a function?", "A reusable block of code that performs a task."),
        ("What is object-oriented programming?", "Organising code into objects that bundle data and behaviour."),
        ("What is a compiler?", "A program that translates source code into machine code."),
        ("What is a syntax error?", "A mistake in the code that breaks the language's grammar rules."),
        ("What is a bug?", "A flaw in a program that causes incorrect behaviour."),
    ],
    ("Computer Science", "data structures"): [
        ("What is a stack?", "A last-in, first-out data structure."),
        ("What is a queue?", "A first-in, first-out data structure."),
        ("What is a hash table?", "A structure mapping keys to values using a hash function."),
        ("What is a binary search tree?", "A tree where left children are smaller and right children larger."),
        ("What is a linked list?", "Nodes that each point to the next node."),
        ("What is the time complexity of binary search?", "O(log n)."),
        ("What is a graph?", "A set of vertices connected by edges."),
        ("What does a heap guarantee?", "The parent is always ordered before its children."),
    ],
    ("Economics", "microeconomics"): [
        ("What is the law of demand?", "As price rises, quantity demanded falls."),
        ("What is opportunity cost?", "The value of the next best alternative given up."),
        ("What is price elasticity of demand?", "How much quantity demanded responds to a price change."),
        ("What is a monopoly?", "A market with a single seller and no close substitutes."),
        ("What is marginal cost?", "The cost of producing one more unit."),
        ("What is market equilibrium?", "The price where quantity supplied equals quantity demanded."),
        ("What is a public good?", "A good that is non-excludable and non-rivalrous."),
        ("What causes a shift in the supply curve?", "A change in production costs or technology."),
    ],
    ("Economics", "macroeconomics"): [
        ("What is GDP?", "The total value of goods and services produced in a country."),
        ("What is inflation?", "A sustained rise in the general price level."),
        ("What does a central bank do?", "Sets interest rates and manages the money supply."),
        ("What is fiscal policy?", "Government spending and taxation decisions."),
        ("What is unemployment?", "People who want work and are actively seeking it but have none."),
        ("What is a recession?", "A significant decline in economic activity lasting months."),
        ("What is monetary policy?", "Controlling the money supply and interest rates to steer the economy."),
        ("What is a trade deficit?", "When imports exceed exports."),
    ],
}

cards = [(deck, q, a) for (deck, _), qas in TOPICS.items() for q, a in qas]
build_apkg(sys.argv[1], cards)
print(len(cards), "cards,", len({d for d, _, _ in cards}), "decks ->", sys.argv[1])
