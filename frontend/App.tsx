import React from 'react';
import ChatWidget from './components/ChatWidget';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Navbar Mockup */}
      <nav className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center text-white font-bold text-xl">I</div>
            <span className="font-bold text-xl text-gray-800 tracking-tight">ilaria.ge</span>
          </div>
          <div className="hidden md:flex gap-8 text-sm font-medium text-gray-600">
            <a href="#" className="hover:text-primary transition-colors">მთავარი</a>
            <a href="#" className="hover:text-primary transition-colors">ფუნქციები</a>
            <a href="#" className="hover:text-primary transition-colors">ტარიფები</a>
            <a href="#" className="hover:text-primary transition-colors">კონტაქტი</a>
          </div>
          <button className="bg-gray-900 text-white px-5 py-2 rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors">
            სისტემაში შესვლა
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1">
        <div className="container mx-auto px-6 py-16 md:py-24">
          <div className="max-w-4xl mx-auto text-center space-y-8">
            <h1 className="text-4xl md:text-6xl font-extrabold text-gray-900 leading-tight">
              თანამედროვე <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">HR მენეჯმენტის</span><br />
              სრული გადაწყვეტა
            </h1>
            <p className="text-lg md:text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed">
              მართეთ თანამშრომლები, შვებულებები და სახელფასო უწყისები ერთ სივრცეში. 
              მარტივი, სწრაფი და უსაფრთხო სისტემა თქვენი ბიზნესისთვის.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <button className="bg-primary text-white px-8 py-4 rounded-xl text-lg font-semibold hover:bg-primary/90 transition-all shadow-lg shadow-primary/30">
                დაიწყე უფასოდ
              </button>
              <button className="bg-white text-gray-700 border border-gray-200 px-8 py-4 rounded-xl text-lg font-semibold hover:bg-gray-50 transition-all">
                დემო ვერსია
              </button>
            </div>
          </div>

          {/* Features Grid Mockup */}
          <div className="grid md:grid-cols-3 gap-8 mt-24">
            {[
              { title: "თანამშრომლები", icon: "👥", desc: "თანამშრომელთა პირადი საქმეების ციფრული არქივი" },
              { title: "შვებულებები", icon: "✈️", desc: "შვებულებებისა და ბიულეტენების ავტომატური აღრიცხვა" },
              { title: "რეპორტინგი", icon: "📊", desc: "დეტალური ანალიტიკა და რეპორტები HR პროცესებზე" }
            ].map((feature, idx) => (
              <div key={idx} className="bg-white p-8 rounded-2xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
                <div className="w-12 h-12 bg-gray-50 rounded-xl flex items-center justify-center text-2xl mb-6">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-500 leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>

          {/* Demo Indicator */}
          <div className="mt-20 text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 text-blue-700 rounded-full text-sm font-medium">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></span>
              </span>
              ჩატბოტი აქტიურია
            </div>
            <p className="mt-4 text-gray-500 text-sm">
              დააჭირეთ მარჯვენა ქვედა კუთხეში არსებულ ღილაკს, რომ გაესაუბროთ Ilaria AI-ს
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-100 py-12">
        <div className="container mx-auto px-6 text-center text-gray-400 text-sm">
          <p>&copy; 2024 Ilaria HR System. ყველა უფლება დაცულია.</p>
        </div>
      </footer>

      {/* The Chat Widget Integration */}
      <ChatWidget />
    </div>
  );
}

export default App;