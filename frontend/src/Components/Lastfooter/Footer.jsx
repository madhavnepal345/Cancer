import { motion } from "framer-motion";
import { FaInstagram, FaFacebookF, FaTwitter } from "react-icons/fa";
import Logo from "../../Images/logo.png"; // replace with your logo path

export default function Footer() {
  return (
    <footer className="bg-gray-800 text-white py-12 px-4">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-8">

        {/* Logo & Description */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="flex flex-col items-start"
        >
          <img
            src={Logo}
            alt="Babble Body Logo"
            className="w-24 h-24 mb-4 rounded-full object-cover border-2 border-pink-400"
          />
          <p className="text-white max-w-xs">
            Babble Body is your AI companion for breast health awareness, guidance, and support.
          </p>
        </motion.div>

        {/* Links */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          viewport={{ once: true }}
          className="flex flex-col space-y-2"
        >
          <h3 className="font-semibold text-pink-600 mb-2">Quick Links</h3>
          <a href="#" className="hover:text-pink-500 transition-colors">Home</a>
          <a href="#" className="hover:text-pink-500 transition-colors">About</a>
          <a href="#" className="hover:text-pink-500 transition-colors">FAQ</a>
          <a href="#" className="hover:text-pink-500 transition-colors">Contact</a>
          <a href="#" className="hover:text-pink-500 transition-colors">Privacy Policy</a>
        </motion.div>

        {/* Social Media */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          viewport={{ once: true }}
          className="flex flex-col items-start space-y-2"
        >
          <h3 className="font-semibold text-pink-600 mb-2">Follow Us</h3>
          <div className="flex gap-4 text-white text-xl">
            <a href="#" className="hover:text-pink-500 transition-colors"><FaInstagram /></a>
            <a href="#" className="hover:text-pink-500 transition-colors"><FaFacebookF /></a>
            <a href="#" className="hover:text-pink-500 transition-colors"><FaTwitter /></a>
          </div>
        </motion.div>
      </div>

      {/* Copyright */}
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.6 }}
        viewport={{ once: true }}
        className="mt-8 text-center text-white text-sm"
      >
        © 2025 Babble Body. All rights reserved.
      </motion.div>
    </footer>
  );
}
