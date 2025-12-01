import { motion } from "framer-motion";
import { MessageCircle, BookOpen, Bell } from "lucide-react"; // icons

export default function HowItWorks() {
  const steps = [
    {
      icon: <MessageCircle size={40} className="text-pink-500" />,
      title: "Ask Questions",
      description: "Chat anytime with AI for guidance.",
    },
    {
      icon: <BookOpen size={40} className="text-pink-500" />,
      title: "Get Reliable Info",
      description: "Evidence-based answers on breast health.",
    },
    {
      icon: <Bell size={40} className="text-pink-500" />,
      title: "Stay Updated",
      description: "Tips, alerts, and awareness resources.",
    },
  ];

  return (
    <section className="py-16 bg-white  rounded-2xl" name="features">
      {/* Heading */}
      <motion.h2
        className="text-3xl md:text-4xl font-bold text-center text-pink-700 mb-12"
        initial={{ opacity: 0, y: -30 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        viewport={{ once: true }}
      >
        How Babble Body Helps You
      </motion.h2>

      {/* Cards Container */}
      <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row gap-8">
        {steps.map((step, index) => (
          <motion.div
            key={index}
            className="bg-white rounded-2xl cursor-pointer shadow-lg p-6 flex flex-col items-center text-center hover:shadow-xl transition-shadow duration-300"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.2, duration: 0.6 }}
            viewport={{ once: true }}
            whileHover={{ scale: 1.05 }}
          >
            {step.icon}
            <h3 className="text-xl font-semibold text-gray-800 mt-4">{step.title}</h3>
            <p className="text-gray-600 mt-2">{step.description}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
