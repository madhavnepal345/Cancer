import { motion } from "framer-motion";

const testimonials = [
  {
    name: "Emily R.",
    role: "Patient",
    quote: "Babble Body gave me the guidance I needed. The AI is friendly and informative!",
    image: "https://randomuser.me/api/portraits/women/44.jpg",
  },
  {
    name: "Dr. Sarah K.",
    role: "Healthcare Professional",
    quote: "A reliable tool for patients to get immediate guidance and support.",
    image: "https://randomuser.me/api/portraits/women/68.jpg",
  },
  {
    name: "Anita P.",
    role: "Patient",
    quote: "Using Babble Body made me feel supported and informed. Highly recommend it!",
    image: "https://randomuser.me/api/portraits/women/22.jpg",
  },
];

export default function Testimonials() {
  return (
    <section className="bg-white py-16 px-4">
      <h2 className="text-3xl font-bold text-center mb-12 text-pink-700">
        What Users Say
      </h2>

      <div className="flex flex-wrap justify-center gap-8">
        {testimonials.map((t, index) => (
          <motion.div
            key={index}
            className="bg-white rounded-xl shadow-md p-6 max-w-xs text-center"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: index * 0.2 }}
            viewport={{ once: true }}
          >
            <img
              src={t.image}
              alt={t.name}
              className="w-24 h-24 mx-auto rounded-full mb-4 border-4 border-pink-200"
            />
            <p className="text-gray-700 mb-2">"{t.quote}"</p>
            <h3 className="font-semibold text-pink-700">{t.name}</h3>
            <span className="text-gray-500 text-sm">{t.role}</span>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
