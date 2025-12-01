import { motion } from "framer-motion";
import { FaKeyboard, FaRobot, FaComments, FaHeart } from "react-icons/fa";

const steps = [
  {
    id: 1,
    title: "Ask Your Question",
    description: "Type your medical question or symptom in the chat box.",
    icon: <FaKeyboard className="text-pink-400 w-10 h-10" />,
  },
  {
    id: 2,
    title: "AI Analyzes Your Query",
    description: "The chatbot processes your question using AI for accurate guidance.",
    icon: <FaRobot className="text-pink-400 w-10 h-10" />,
  },
  {
    id: 3,
    title: "Get Instant Guidance",
    description: "Receive answers with remedies, precautions, or advice to see a doctor.",
    icon: <FaComments className="text-pink-400 w-10 h-10" />,
  },
  {
    id: 4,
    title: "Follow Up",
    description: "Ask follow-up questions until all your doubts are clarified.",
    icon: <FaHeart className="text-pink-400 w-10 h-10" />,
  },
];

export default function HowItWorks() {
  return (
    <section className="py-20 " name="howit'swork">
      <div className="max-w-6xl mx-auto px-4 text-center">
        <motion.h2
          className="text-3xl md:text-4xl font-bold text-pink-700 mb-4"
          initial={{ opacity: 0, y: -40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          viewport={{ once: true }}
        >
          How It Works
        </motion.h2>
        <motion.p
          className="text-gray-600 mb-12"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          viewport={{ once: true }}
        >
          Babble Body helps you get medical guidance in 4 simple steps.
        </motion.p>

        <div className="grid md:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <motion.div
              key={step.id}
              className="bg-white p-6 rounded-3xl shadow-lg cursor-pointer"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              whileHover={{ y: -10, boxShadow: "0 15px 35px rgba(236, 72, 153, 0.3)" }}
              transition={{ duration: 0.5, delay: index * 0.2, type: "spring", stiffness: 120 }}
              viewport={{ once: true }}
            >
              <div className="mb-4 flex justify-center">{step.icon}</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">{step.title}</h3>
              <p className="text-gray-600">{step.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
