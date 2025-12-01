import { motion } from "framer-motion";
import LandingPageImage from "../../Images/landing_page.jpg"; // Background image
import ChatbotImage from "../../Images/chatBot2.webp"; // ✅ Your AI character PNG
import { useNavigate } from "react-router-dom";

export default function LandingPage() {

  const navigate = useNavigate();

  return (
    <section className="bg-white min-h-screen flex items-center justify-center px-6 md:px-16 py-12" name="home">
      <div className="max-w-7xl w-full grid md:grid-cols-2 gap-12 items-center">
        
        {/* Image Section (on mobile, appears first) */}
        <motion.div
          initial={{ opacity: 0, x: 60, rotate: 5 }}
          animate={{ opacity: 1, x: 0, rotate: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
          className="relative flex justify-center order-1 md:order-2"
        >
          {/* Floating Decorative Shape */}
          

          {/* Image */}
          <img
            src={ChatbotImage} // Replace with your chatbot image
            alt="OncoBot Chatbot"
            className="rounded-3xl  w-full max-w-md object-cover"
          />
        </motion.div>

        {/* Text Section */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-center md:text-left order-2 md:order-1"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 leading-snug">
            Artificial Intelligence for <br /> Cancer-Aware
          </h1>
          <p className="mt-6 text-lg text-gray-600">
            OncoBot is your friendly AI Medical Assistant designed to to answer cancer-related questions with high reliability.
          </p>
          <motion.button
          onClick={()=>navigate('/chat')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.97 }}
            className="mt-8 px-6 py-3 bg-pink-500 hover:bg-pink-400 cursor-pointer text-white font-semibold rounded-full shadow-md transition"
          >
            Start Chat Now
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}
