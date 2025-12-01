import { useState } from "react";
import { motion } from "framer-motion";
import { Menu, X } from "lucide-react";
import logo from "../../Images/logo.PNG"
import { Link } from "react-scroll";
import { useNavigate } from "react-router-dom";


export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
   const navigate = useNavigate();
  const navLinks = [
    {
      id: 1,
      title: "Home",
    },
    {
      id: 2,
      title: "How It's Work",
    },
    {
      id: 3,
      title: "FAQs",
    }
    
  ]

  return (
    <nav className="bg-pink-100 shadow-md fixed w-full z-50 rounded-b-lg py-2">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-18">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="font-bold text-xl text-gray-800"
          >
            <img text="OncoBot" alt="Babble Body Logo" className="h-16 rounded-full w-auto mr-2" />
          </motion.div>

          {/* Desktop Nav */}
          <div className="hidden md:flex space-x-20 font-medium text-[17px]">
      {navLinks.map((link) => (
        <motion.div
          key={link.id}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.97 }}
        >
          <Link
            to={link.title.replace(/\s+/g, "").toLowerCase()} // example: "How It's Work" → "howit'swork"
            smooth={true}
            duration={500}
            offset={-70} // adjust for fixed navbar height
            className="cursor-pointer text-black hover:text-pink-500 transition"
          >
            {link.title}
          </Link>
        </motion.div>
      ))}
    </div>

          {/* Chat Button */}
          <motion.button
          onClick={() => navigate('/chat')}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="hidden md:block bg-pink-500 hover:bg-pink-400 text-white px-6 py-2 rounded-lg shadow-md cursor-pointer"
            
          >
            Chat
          </motion.button>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button onClick={() => setIsOpen(!isOpen)}>
              {isOpen ? <X size={28} /> : <Menu size={28} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
  <motion.div
    initial={{ height: 0, opacity: 0 }}
    animate={{ height: "auto", opacity: 1 }}
    transition={{ duration: 0.3 }}
    className="md:hidden bg-pink-100 px-4 pb-4"
  >
    {navLinks.map((link, index) => (
      <motion.div
        key={link.id}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: index * 0.1 }}
      >
        <Link
          to={link.title.replace(/\s+/g, "").toLowerCase()}
          smooth={true}
          duration={500}
          offset={-70}
          onClick={() => setIsOpen(false)}   // close menu after click
          className="block py-2 text-gray-700 hover:text-pink-500 cursor-pointer"
        >
          {link.title}
        </Link>
      </motion.div>
    ))}

    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => navigate("/chat")}
      className="mt-2 w-full bg-pink-400 text-white py-2 rounded-lg shadow-md"
    >
      Chat
    </motion.button>
  </motion.div>
)}

    </nav>
  );
}
