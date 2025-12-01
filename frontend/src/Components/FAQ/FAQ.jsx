import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const faqs = [
  {
    question: "Is Babble Body a substitute for medical advice?",
    answer:
      "No. Babble Body provides guidance and educational information, but it is not a replacement for professional medical advice.",
  },
  {
    question: "Is my data safe while using the chatbot?",
    answer:
      "Yes. We prioritize your privacy and ensure all conversations are secure and confidential.",
  },
  {
    question: "Can I ask questions about symptoms?",
    answer:
      "Yes. The AI chatbot can provide information on symptoms, but always consult a healthcare professional for diagnosis.",
  },
  {
    question: "How often is the chatbot information updated?",
    answer:
      "Our AI is regularly updated with the latest research and verified medical sources.",
  },
];

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState(null);

  const toggleFAQ = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section className="bg-white py-16 px-4" name="faqs">
      <h2 className="text-3xl font-bold text-center text-pink-700 mb-12">
        Frequently Asked Questions
      </h2>

      <div className="max-w-3xl mx-auto space-y-4">
        {faqs.map((faq, index) => (
          <div key={index} className="border rounded-lg overflow-hidden shadow-sm">
            <button
              onClick={() => toggleFAQ(index)}
              className="w-full text-left px-6 py-4 bg-pink-50 flex justify-between items-center focus:outline-none"
            >
              <span className="font-medium text-pink-700">{faq.question}</span>
              <span className="text-pink-500 font-bold">
                {openIndex === index ? "-" : "+"}
              </span>
            </button>

            <AnimatePresence>
              {openIndex === index && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="px-6 py-4 bg-white text-gray-700"
                >
                  {faq.answer}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </section>
  );
}
