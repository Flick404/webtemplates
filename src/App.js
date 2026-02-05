import { useState } from 'react';
import './App.css';

function App() {
  const [active, setActive] = useState('salon');

  const renderPage = () => {
    switch (active) {
      case 'salon':
        return <SalonPage />;
      case 'emergency':
        return <EmergencyPage />;
      case 'pet':
        return <PetPage />;
      case 'cafe':
        return <CafePage />;
      case 'tutoring':
        return <TutoringPage />;
      default:
        return <SalonPage />;
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-inner">
          <div className="brand">
            <span className="brand-dot" />
            <span>
              LANDINGLAB
              <span className="brand-sub">Service landing page explorer</span>
            </span>
          </div>
          <nav className="nav" aria-label="Choose example website">
            <NavButton
              label="Hair Salon"
              isActive={active === 'salon'}
              onClick={() => setActive('salon')}
            />
            <NavButton
              label="Plumber / Electrician"
              isActive={active === 'emergency'}
              onClick={() => setActive('emergency')}
            />
            <NavButton
              label="Pet Groomer"
              isActive={active === 'pet'}
              onClick={() => setActive('pet')}
            />
            <NavButton
              label="Café / Restaurant"
              isActive={active === 'cafe'}
              onClick={() => setActive('cafe')}
            />
            <NavButton
              label="Tutoring Center"
              isActive={active === 'tutoring'}
              onClick={() => setActive('tutoring')}
            />
          </nav>
        </div>
      </header>
      <main className="app-main">{renderPage()}</main>
    </div>
  );
}

function NavButton({ label, isActive, onClick }) {
  return (
    <button
      type="button"
      className={`nav-btn ${isActive ? 'is-active' : ''}`}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

function SalonPage() {
  return (
    <div className="page page-salon">
      <div className="page-inner">
        <section className="hero">
          <div className="hero-layout">
            <div>
              <p className="hero-kicker">Eco-chic hair salon • [City]</p>
              <h1 className="hero-title">
                Fresh cuts, soft color, and a salon that&apos;s easy to love
              </h1>
              <p className="hero-subtitle">
                Modern cuts, organic color and a calm, light-filled space.
                Everything is designed so you walk out feeling like yourself—
                just better.
              </p>
              <p className="hero-meta">
                <span className="hero-pill">
                  <span>4.9★</span>
                  <span>1,200+ reviews in [City]</span>
                </span>
              </p>
            </div>
            <aside className="hero-aside">
              <div className="hero-media">
                <img
                  className="hero-media-img"
                  src="https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?auto=format&fit=crop&w=900&q=80"
                  alt="Stylist blow-drying a client in a modern hair salon"
                />
              </div>
              <div className="info-card">
                <h3>At a glance</h3>
                <p>Specialists in natural color, precise cuts, and healthy hair.</p>
                <div className="stats">
                  <div className="stat">
                    <div className="stat-label">Experience</div>
                    <div className="stat-value">10+ years</div>
                  </div>
                  <div className="stat">
                    <div className="stat-label">Approach</div>
                    <div className="stat-value">Organic, gentle care</div>
                  </div>
                  <div className="stat">
                    <div className="stat-label">Clients</div>
                    <div className="stat-value">Trusted across [City]</div>
                  </div>
                </div>
              </div>
            </aside>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>Services that fit your everyday life</h2>
            <p>
              From precision cuts to vibrant color transformations, we offer
              everything you need to look and feel your best.
            </p>
          </header>
          <div className="grid grid-3">
            <div className="card card--highlight">
              <h3>Precision cuts</h3>
              <p>Wash, cut and finish tailored to your hair type and routine.</p>
              <ul className="list">
                <li>Short, medium and long hair</li>
                <li>Fringe trims and reshaping</li>
                <li>Consultation included</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Color &amp; highlights</h3>
              <p>Soft blondes, rich brunettes and lived-in color that grows well.</p>
              <ul className="list">
                <li>Balayage and foils</li>
                <li>Toners and glossing</li>
                <li>Color correction</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Barber services</h3>
              <p>Clean fades, beard shaping and classic scissor cuts.</p>
              <ul className="list">
                <li>Beard trims and line-ups</li>
                <li>Hot towel finishes</li>
                <li>Traditional shaves</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header section-header--center">
            <h2>What makes us different</h2>
            <p>We&apos;re not just another salon—here&apos;s what sets us apart</p>
          </header>
          <div className="grid grid-4">
            <div className="card card--feature">
              <div className="card--feature-icon">🌿</div>
              <h3>Eco-friendly</h3>
              <p>100% organic, cruelty-free products that care for your hair and the planet.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">✂️</div>
              <h3>Expert stylists</h3>
              <p>Certified professionals with years of experience and ongoing training.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">💆</div>
              <h3>Relaxed atmosphere</h3>
              <p>Calm, light-filled space designed for comfort and peace of mind.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">⭐</div>
              <h3>Proven results</h3>
              <p>4.9★ rating with over 1,200 happy clients across [City].</p>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header section-header--center">
            <h2>Real results from real clients</h2>
            <p>See the transformations that keep our clients coming back</p>
          </header>
          <div className="gallery">
            <figure className="gallery-item">
              <img
                className="gallery-img"
                src="https://images.unsplash.com/photo-1519741497674-611481863552?auto=format&fit=crop&w=800&q=80"
                alt="Woman with styled long hair"
              />
              <figcaption className="gallery-caption">
                Soft balayage and layered cut for everyday wear
              </figcaption>
            </figure>
            <figure className="gallery-item">
              <img
                className="gallery-img"
                src="https://images.unsplash.com/photo-1517832606299-7ae9b720a186?auto=format&fit=crop&w=800&q=80"
                alt="Barber giving a precise fade haircut"
              />
              <figcaption className="gallery-caption">
                Skin fade &amp; beard shape-up with razor detailing
              </figcaption>
            </figure>
            <figure className="gallery-item">
              <img
                className="gallery-img"
                src="https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?auto=format&fit=crop&w=800&q=80"
                alt="Stylist blow-drying hair"
              />
              <figcaption className="gallery-caption">
                Volume blowout finished with nourishing products
              </figcaption>
            </figure>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>Why regulars keep coming back</h2>
          </header>
          <div className="grid grid-2">
            <div>
              <p>
                The best salon experiences feel relaxed, not rushed. We book with
                enough time to consult, cut and style so people leave feeling
                looked after. Our stylists take time to understand your hair and
                lifestyle.
              </p>
              <div className="stats">
                <div className="stat">
                  <div className="stat-label">Retention</div>
                  <div className="stat-value">80% returning clients</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Care</div>
                  <div className="stat-value">Cruelty‑free products</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Satisfaction</div>
                  <div className="stat-value">4.9★ average rating</div>
                </div>
              </div>
            </div>
            <div>
              <div className="testimonial">
                <p>
                  &ldquo;They really listen, explain what will work with my hair,
                  and I never feel pressured into something I don&apos;t want.
                  Best salon experience I&apos;ve had.&rdquo;
                </p>
                <span>— Laura M., [Neighborhood]</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;My color grows out soft, no harsh lines. People ask where
                  I go all the time. The team is so talented and friendly.&rdquo;
                </p>
                <span>— Amira K., [City]</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;The only place I trust for my fades. Always sharp,
                  always on time, and they really know how to work with my hair
                  texture.&rdquo;
                </p>
                <span>— Diego S., [City]</span>
              </div>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>How it works</h2>
            <p>Simple steps to your perfect look</p>
          </header>
          <div className="process-container">
            <div className="process-step">
              <h3>Book your appointment</h3>
              <p>Choose a time that works for you. We offer flexible scheduling including evenings and weekends.</p>
            </div>
            <div className="process-step">
              <h3>Consultation</h3>
              <p>We start with a chat about your hair goals, lifestyle, and what you&apos;re looking for.</p>
            </div>
            <div className="process-step">
              <h3>Service &amp; style</h3>
              <p>Our stylists work their magic while you relax. We take our time to get it just right.</p>
            </div>
            <div className="process-step">
              <h3>Walk out confident</h3>
              <p>Leave feeling refreshed, looking great, and with tips to maintain your new style at home.</p>
            </div>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>Frequently asked questions</h2>
          </header>
          <div>
            <div className="faq-item">
              <div className="faq-question">How far in advance should I book?</div>
              <div className="faq-answer">We recommend booking 2-3 weeks ahead, especially for weekends. We do accept same-day appointments when available.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">Do you use organic products?</div>
              <div className="faq-answer">Yes! All our products are 100% organic and cruelty-free. We&apos;re committed to using only the best for your hair and the environment.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">What if I&apos;m not happy with my cut?</div>
              <div className="faq-answer">We stand behind our work. If you&apos;re not completely satisfied, come back within 7 days and we&apos;ll fix it at no charge.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">Do you offer gift cards?</div>
              <div className="faq-answer">Absolutely! Gift cards are available in any amount and make perfect gifts for birthdays, holidays, or just because.</div>
            </div>
          </div>
        </section>

        <section className="section section--band">
          <div className="section-header section-header--center">
            <h2>Ready to transform your look?</h2>
            <p>Join over 1,200 happy clients who trust us with their hair</p>
          </div>
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <p style={{ fontSize: '1.1rem', marginBottom: '1rem' }}>
              Call us at <a href="tel:+34123456789" style={{ color: '#ffffff', fontWeight: '700' }}>+34 123 456 789</a> or visit us at [Street address]
            </p>
          </div>
        </section>

        <footer className="footer">
          <div>
            <h4>Location</h4>
            <p>[Street address]</p>
            <p>[City]</p>
          </div>
          <div>
            <h4>Hours</h4>
            <p>Tue–Fri: 10:00–19:00</p>
            <p>Sat: 9:00–17:00</p>
            <p>Sun–Mon: Closed</p>
          </div>
          <div>
            <h4>Contact</h4>
            <p>Phone: <a href="tel:+34123456789">+34 123 456 789</a></p>
            <p>Email: hello@salonexample.com</p>
          </div>
        </footer>
      </div>
    </div>
  );
}

function EmergencyPage() {
  return (
    <div className="page page-emergency">
      <div className="page-inner">
        <section className="hero">
          <div className="hero-layout">
            <div>
              <p className="hero-kicker">Plumbing • Electrical • Locksmith</p>
              <h1 className="hero-title">
                Reliable emergency help in [City], day or night
              </h1>
              <p className="hero-subtitle">
                A small, local team of licensed technicians for leaks, power
                issues and lockouts. Clear explanations and no surprise fees.
              </p>
              <p className="hero-meta">
                <span className="hero-pill">
                  <span>24/7</span>
                  <span>Technicians on call across [City]</span>
                </span>
              </p>
            </div>
            <aside className="hero-aside">
              <div className="hero-media">
                <img
                  className="hero-media-img"
                  src="https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?auto=format&fit=crop&w=900&q=80"
                  alt="Technician working on a bathroom repair"
                />
              </div>
              <div className="info-card">
                <h3>Typical response</h3>
                <p>Most urgent jobs in [City] reached within 30–45 minutes.</p>
                <ul className="list">
                  <li>Licensed &amp; insured team</li>
                  <li>Upfront pricing before work starts</li>
                  <li>Workmanship guarantee on every job</li>
                  <li>No call-out fees</li>
                </ul>
              </div>
            </aside>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>What we handle most often</h2>
            <p>
              From burst pipes to power outages, we&apos;re equipped to handle
              your emergency quickly and professionally.
            </p>
          </header>
          <div className="grid grid-3">
            <div className="card card--highlight">
              <h3>Plumbing</h3>
              <p>Emergencies and small fixes for homes and apartments.</p>
              <ul className="list">
                <li>Leaks and burst pipes</li>
                <li>Clogged drains and toilets</li>
                <li>Water heater issues</li>
                <li>Fixture repairs</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Electrical</h3>
              <p>Safe repairs for outages, tripping breakers and wiring.</p>
              <ul className="list">
                <li>No‑power troubleshooting</li>
                <li>Breaker and panel issues</li>
                <li>Outlet and lighting faults</li>
                <li>Safety inspections</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Locksmith</h3>
              <p>Non‑destructive entry wherever possible, day or night.</p>
              <ul className="list">
                <li>Locked out of home or office</li>
                <li>Lock changes and rekeying</li>
                <li>Basic key duplication</li>
                <li>Security upgrades</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header section-header--center">
            <h2>Why choose us</h2>
            <p>We&apos;re not just another service—here&apos;s what makes us different</p>
          </header>
          <div className="grid grid-4">
            <div className="card card--feature">
              <div className="card--feature-icon">⚡</div>
              <h3>Fast response</h3>
              <p>Average 30-45 minute arrival time for urgent emergencies across [City].</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">🔒</div>
              <h3>Licensed &amp; insured</h3>
              <p>Fully certified technicians with proper insurance for your peace of mind.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">💰</div>
              <h3>Upfront pricing</h3>
              <p>No surprises—we quote before we start, so you know exactly what you&apos;re paying.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">⭐</div>
              <h3>Proven track record</h3>
              <p>4.8★ rating with 200+ reviews from satisfied customers across [City].</p>
            </div>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>Trusted by households across [City]</h2>
          </header>
          <div className="grid grid-2">
            <div>
              <p>
                Emergencies are stressful, so we keep communication simple:
                explain the issue, share options, agree the price, then fix it.
                No upselling, no surprises—just honest, reliable service.
              </p>
              <div className="badge-row">
                <span className="badge-soft">Licensed &amp; insured</span>
                <span className="badge-soft">Local team, no call centres</span>
                <span className="badge-soft">Upfront pricing</span>
                <span className="badge-soft">15+ years experience</span>
              </div>
            </div>
            <div>
              <div className="testimonial">
                <p>
                  &ldquo;Water heater leak at 11pm. They arrived quickly, shut
                  everything off and had us safe until a full repair the next
                  morning. Professional and reassuring.&rdquo;
                </p>
                <span>— Jane D., Central [City]</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;Locked out with kids. They were calm, polite and had us
                  back inside without drilling the lock. Worth every penny.&rdquo;
                </p>
                <span>— Mark R., North [City]</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;Power went out during dinner. They diagnosed and fixed
                  the breaker issue in under an hour. Great service!&rdquo;
                </p>
                <span>— Sarah L., East [City]</span>
              </div>
              <p className="hero-meta">
                <span className="pill-rating">
                  <strong>4.8★</strong>
                  <span>Google • 200+ reviews</span>
                </span>
              </p>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>How we work</h2>
            <p>Simple, transparent process from call to completion</p>
          </header>
          <div className="process-container">
            <div className="process-step">
              <h3>You call us</h3>
              <p>24/7 emergency line. Describe your issue and we&apos;ll give you an estimated arrival time.</p>
            </div>
            <div className="process-step">
              <h3>We arrive fast</h3>
              <p>Most urgent calls reached within 30-45 minutes. We come fully equipped to handle the job.</p>
            </div>
            <div className="process-step">
              <h3>We assess &amp; quote</h3>
              <p>We explain what&apos;s wrong, show you options, and give you an upfront price before starting.</p>
            </div>
            <div className="process-step">
              <h3>We fix it right</h3>
              <p>Quality work with a guarantee. We clean up after ourselves and make sure everything works.</p>
            </div>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>Common questions</h2>
          </header>
          <div>
            <div className="faq-item">
              <div className="faq-question">Do you charge a call-out fee?</div>
              <div className="faq-answer">No, we don&apos;t charge call-out fees. You only pay for the work we do, and we quote upfront so there are no surprises.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">Are your technicians licensed?</div>
              <div className="faq-answer">Yes, all our technicians are fully licensed and insured. We&apos;re certified for plumbing, electrical, and locksmith services.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">What areas do you serve?</div>
              <div className="faq-answer">We cover all of [City] including Central, North, East, and West neighborhoods. We also serve surrounding areas—call to confirm.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">Do you offer warranties?</div>
              <div className="faq-answer">Yes, all our work comes with a workmanship guarantee. If something goes wrong due to our work, we&apos;ll fix it at no charge.</div>
            </div>
          </div>
        </section>

        <section className="section section--band">
          <div className="section-header section-header--center">
            <h2>Need help right now?</h2>
            <p>Our technicians are standing by 24/7 to assist you</p>
          </div>
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <p style={{ fontSize: '1.1rem', marginBottom: '1rem' }}>
              Call <a href="tel:+34123456789" style={{ color: '#ffffff', fontWeight: '700' }}>+34 123 456 789</a> for immediate assistance
            </p>
          </div>
        </section>

        <footer className="footer">
          <div>
            <h4>Service area</h4>
            <p>Central, North, East &amp; West [City]</p>
            <p>Residential and small commercial</p>
          </div>
          <div>
            <h4>When we work</h4>
            <p>Emergency line: 24/7</p>
            <p>Office hours: Mon–Fri 8:00–18:00</p>
          </div>
          <div>
            <h4>Contact</h4>
            <p>Phone: <a href="tel:+34123456789">+34 123 456 789</a></p>
            <p>Email: help@fastfixexample.com</p>
          </div>
        </footer>
      </div>
    </div>
  );
}

function PetPage() {
  return (
    <div className="page page-pet">
      <div className="page-inner">
        <section className="hero">
          <div className="hero-layout">
            <div>
              <p className="hero-kicker">Pet grooming &amp; daycare • [City]</p>
              <h1 className="hero-title">
                Gentle grooming and safe play for your furry family
              </h1>
              <p className="hero-subtitle">
                A bright, calm space where dogs and cats are handled with
                patience. Clear routines, small groups and lots of updates for
                owners.
              </p>
              <p className="hero-meta">
                <span className="hero-pill">
                  <span>100+ ★★★★★</span>
                  <span>Reviews from local pet parents</span>
                </span>
              </p>
            </div>
            <aside className="hero-aside">
              <div className="hero-media">
                <img
                  className="hero-media-img"
                  src="https://images.unsplash.com/photo-1548199973-03cce0bbc87b?auto=format&fit=crop&w=900&q=80"
                  alt="Happy groomed dog with a bandana"
                />
              </div>
              <div className="info-card">
                <h3>Quick facts</h3>
                <ul className="list">
                  <li>Certified groomers and pet‑first handling</li>
                  <li>Pet CPR &amp; first‑aid trained staff</li>
                  <li>Cage‑free options for suitable dogs</li>
                  <li>Photo updates during visits</li>
                </ul>
              </div>
            </aside>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>What we offer</h2>
            <p>
              Comprehensive care services designed to keep your pets happy,
              healthy, and looking their best.
            </p>
          </header>
          <div className="grid grid-3">
            <div className="card card--highlight">
              <h3>Grooming</h3>
              <p>Baths, full grooms and tidy‑ups for all coat types.</p>
              <ul className="list">
                <li>Breed‑aware cuts</li>
                <li>Nail trims and ear cleaning</li>
                <li>De-shedding treatments</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Daycare</h3>
              <p>Structured play and rest in small, supervised groups.</p>
              <ul className="list">
                <li>Separate spaces by size and energy level</li>
                <li>Photo updates during the day</li>
                <li>Puppy socialization programs</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Spa add‑ons</h3>
              <p>Extras for pets that enjoy a little more pampering.</p>
              <ul className="list">
                <li>Deshedding and conditioning</li>
                <li>Teeth and paw treatments</li>
                <li>Blueberry facials</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header section-header--center">
            <h2>Why pet parents trust us</h2>
            <p>We go above and beyond to ensure your pet&apos;s comfort and safety</p>
          </header>
          <div className="grid grid-4">
            <div className="card card--feature">
              <div className="card--feature-icon">🐾</div>
              <h3>Gentle handling</h3>
              <p>Our team is trained in low-stress handling techniques to keep pets calm and comfortable.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">🏥</div>
              <h3>Pet CPR certified</h3>
              <p>All staff are trained in pet first aid and CPR for your peace of mind.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">📸</div>
              <h3>Photo updates</h3>
              <p>We send photos throughout the day so you can see your pet is happy and having fun.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">❤️</div>
              <h3>Cage-free options</h3>
              <p>For suitable dogs, we offer cage-free daycare in supervised, safe environments.</p>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header section-header--center">
            <h2>Happy pets, happy owners</h2>
            <p>See the transformations that make pet parents smile</p>
          </header>
          <div className="gallery">
            <figure className="gallery-item">
              <img
                className="gallery-img"
                src="https://images.unsplash.com/photo-1548199973-03cce0bbc87b?auto=format&fit=crop&w=800&q=80"
                alt="Fluffy dog freshly groomed with a bandana"
              />
              <figcaption className="gallery-caption">
                Fresh groom, soft coat, and a new bandana to show off
              </figcaption>
            </figure>
            <figure className="gallery-item">
              <img
                className="gallery-img"
                src="https://images.unsplash.com/photo-1537151608828-ea2b11777ee8?auto=format&fit=crop&w=800&q=80"
                alt="Dog being dried at the grooming salon"
              />
              <figcaption className="gallery-caption">
                Gentle drying and brushing for a stress-free experience
              </figcaption>
            </figure>
            <figure className="gallery-item">
              <img
                className="gallery-img"
                src="https://images.unsplash.com/photo-1530281700549-e82e7bf110d6?auto=format&fit=crop&w=800&q=80"
                alt="Dog relaxing on a cushion in a pet daycare"
              />
              <figcaption className="gallery-caption">
                Calm, cozy rest areas between supervised play sessions
              </figcaption>
            </figure>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>For anxious or first‑time pets</h2>
          </header>
          <div className="grid grid-2">
            <div>
              <p>
                Many pets feel unsure during their first few visits. We start slowly
                with gentle handling, short sessions and plenty of positive
                reinforcement. Our team is trained to recognize stress signals and
                adjust their approach accordingly.
              </p>
              <div className="stats">
                <div className="stat">
                  <div className="stat-label">Intro sessions</div>
                  <div className="stat-value">Short &amp; low‑pressure</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Updates</div>
                  <div className="stat-value">Messages and photos</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Success rate</div>
                  <div className="stat-value">95% return for regular care</div>
                </div>
              </div>
            </div>
            <div>
              <div className="testimonial">
                <p>
                  &ldquo;Our dog Bella actually gets excited for grooming day. The
                  team is so gentle and patient. She comes home happy and
                  beautiful every time.&rdquo;
                </p>
                <span>— Sarah &amp; Bella, [City]</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;We finally found a daycare where Luna comes home happy
                  and tired, not stressed. The photo updates throughout the day
                  are such a nice touch.&rdquo;
                </p>
                <span>— Carlos &amp; Luna, [City]</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;My anxious cat actually tolerates grooming here. The
                  staff takes their time and never forces anything. Highly
                  recommend!&rdquo;
                </p>
                <span>— Maria &amp; Whiskers, [City]</span>
              </div>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>What to expect</h2>
            <p>A typical visit from start to finish</p>
          </header>
          <div className="process-container">
            <div className="process-step">
              <h3>Welcome &amp; check-in</h3>
              <p>We greet you and your pet warmly, review any special needs or concerns, and answer questions.</p>
            </div>
            <div className="process-step">
              <h3>Gentle introduction</h3>
              <p>For first-time visitors, we take time to let your pet explore and get comfortable with our space.</p>
            </div>
            <div className="process-step">
              <h3>Service with care</h3>
              <p>Whether it&apos;s grooming or daycare, we handle your pet with patience and respect for their comfort.</p>
            </div>
            <div className="process-step">
              <h3>Updates &amp; pickup</h3>
              <p>We send photos during the visit and give you a full report when you pick up your happy, well-cared-for pet.</p>
            </div>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>Common questions</h2>
          </header>
          <div>
            <div className="faq-item">
              <div className="faq-question">What vaccinations are required?</div>
              <div className="faq-answer">Dogs need current rabies, DHPP, and Bordatella. Cats need rabies and FVRCP. We&apos;ll verify records at check-in.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">How long does grooming take?</div>
              <div className="faq-answer">Most full grooms take 2-3 hours depending on size and coat type. We&apos;ll give you an estimated time when you drop off.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">Can I tour the facility?</div>
              <div className="faq-answer">Absolutely! We encourage tours so you can see our clean, safe environment. Call ahead to schedule a convenient time.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">What if my pet is anxious?</div>
              <div className="faq-answer">We specialize in anxious pets! We&apos;ll work at their pace, take breaks as needed, and keep you updated throughout.</div>
            </div>
          </div>
        </section>

        <section className="section section--band">
          <div className="section-header section-header--center">
            <h2>Ready to give your pet the best care?</h2>
            <p>Join 100+ happy pet families who trust us with their furry friends</p>
          </div>
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <p style={{ fontSize: '1.1rem', marginBottom: '1rem' }}>
              Call <a href="tel:+34123456789" style={{ color: '#ffffff', fontWeight: '700' }}>+34 123 456 789</a> or visit us at [Street address]
            </p>
          </div>
        </section>

        <footer className="footer">
          <div>
            <h4>Location</h4>
            <p>[Street address]</p>
            <p>[City]</p>
            <p>Free parking available</p>
          </div>
          <div>
            <h4>Hours</h4>
            <p>Grooming: Mon–Sat 9:00–18:00</p>
            <p>Daycare: Mon–Fri 7:30–19:00</p>
            <p>Sun: Closed</p>
          </div>
          <div>
            <h4>Contact</h4>
            <p>Phone: <a href="tel:+34123456789">+34 123 456 789</a></p>
            <p>Email: hello@petplaceexample.com</p>
          </div>
        </footer>
      </div>
    </div>
  );
}

function CafePage() {
  return (
    <div className="page page-cafe">
      <div className="page-inner">
        <section className="hero">
          <div className="hero-layout">
            <div>
              <p className="hero-kicker">Café &amp; restaurant • [City]</p>
              <h1 className="hero-title">
                A cosy corner for coffee, lunches and long dinners
              </h1>
              <p className="hero-subtitle">
                Warm lighting, simple seasonal food and coffee roasted in the
                neighbourhood. A place that feels familiar even on your first
                visit.
              </p>
              <p className="hero-meta">
                <span className="hero-pill">
                  <span>4.7★</span>
                  <span>Based on 500+ diner reviews</span>
                </span>
              </p>
            </div>
            <aside className="hero-aside">
              <div className="hero-media">
                <img
                  className="hero-media-img"
                  src="https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?auto=format&fit=crop&w=900&q=80"
                  alt="Cozy cafe interior with people enjoying coffee"
                />
              </div>
              <div className="info-card">
                <h3>Today at a glance</h3>
                <ul className="list">
                  <li>Morning: pastries, coffee and light breakfast</li>
                  <li>Afternoon: lunch plates and desserts</li>
                  <li>Evening: shared plates and weekly specials</li>
                  <li>Weekend brunch: 9:00–14:00</li>
                </ul>
              </div>
            </aside>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>What people come back for</h2>
            <p>
              From our signature coffee to seasonal plates, every dish is made
              with care and quality ingredients.
            </p>
          </header>
          <div className="grid grid-3">
            <div className="card card--highlight">
              <h3>Signature coffee</h3>
              <p>
                Beans roasted locally, dialled‑in espresso and simple filter
                options.
              </p>
              <ul className="list">
                <li>Single‑origin espresso</li>
                <li>Oat, soy and dairy options</li>
                <li>Cold brew and specialty drinks</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Seasonal plates</h3>
              <p>Short menus that change with the seasons and local produce.</p>
              <ul className="list">
                <li>Vegetarian and vegan choices</li>
                <li>Comfort dishes with a twist</li>
                <li>Daily specials</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Weekend brunch</h3>
              <p>A relaxed mix of breakfast and lunch favourites.</p>
              <ul className="list">
                <li>Brunch classics and specials</li>
                <li>Fresh juices and spritzes</li>
                <li>Bottomless coffee option</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header section-header--center">
            <h2>What makes us special</h2>
            <p>More than just great food and coffee</p>
          </header>
          <div className="grid grid-4">
            <div className="card card--feature">
              <div className="card--feature-icon">☕</div>
              <h3>Local roasters</h3>
              <p>We partner with neighborhood roasters for the freshest, most flavorful coffee.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">🌱</div>
              <h3>Seasonal ingredients</h3>
              <p>Our menu changes with the seasons, featuring the best local produce available.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">👨‍👩‍👧</div>
              <h3>Family-owned</h3>
              <p>Three generations of recipes and hospitality, making every visit feel like home.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">🏆</div>
              <h3>Award-winning</h3>
              <p>Voted Best Pizza in [City] 2024 and featured in local food guides.</p>
            </div>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>A small story from [City]</h2>
          </header>
          <div className="grid grid-2">
            <div>
              <p>
                Family‑owned for three generations, we focus on simple dishes done
                well. Many of our recipes come from family cookbooks, adjusted over
                time with our team. We source locally when possible and keep our
                menu small so everything can be made fresh.
              </p>
              <div className="badge-row">
                <span className="badge-soft">Family‑run</span>
                <span className="badge-soft">Locally sourced ingredients</span>
                <span className="badge-soft">Walk‑ins welcome</span>
                <span className="badge-soft">Voted Best Pizza 2024</span>
              </div>
            </div>
            <div>
              <div className="testimonial">
                <p>
                  &ldquo;Absolutely charming atmosphere and the risotto is to die
                  for. A must-visit spot in [City]. The staff makes you feel like
                  family.&rdquo;
                </p>
                <span>— Local Guide, [City]</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;The perfect spot for Saturday brunch. Great coffee and
                  friendly staff. We come here every weekend now.&rdquo;
                </p>
                <span>— Ana P., [City]</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;We celebrate every family birthday here. It&apos;s become
                  our tradition. The food is consistently excellent.&rdquo;
                </p>
                <span>— López Family, [City]</span>
              </div>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>Visit us</h2>
            <p>What to expect when you walk through our doors</p>
          </header>
          <div className="process-container">
            <div className="process-step">
              <h3>Find a seat</h3>
              <p>Walk-ins are always welcome. We have cozy tables inside and a lovely patio when weather permits.</p>
            </div>
            <div className="process-step">
              <h3>Order at the counter</h3>
              <p>Check out our daily specials board, then order at the counter. We&apos;ll bring everything to your table.</p>
            </div>
            <div className="process-step">
              <h3>Enjoy fresh food</h3>
              <p>Everything is made to order with fresh ingredients. Expect 15-20 minutes for hot dishes.</p>
            </div>
            <div className="process-step">
              <h3>Stay as long as you like</h3>
              <p>We&apos;re not in a rush. Stay for coffee, work on your laptop, or catch up with friends.</p>
            </div>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>Frequently asked questions</h2>
          </header>
          <div>
            <div className="faq-item">
              <div className="faq-question">Do you take reservations?</div>
              <div className="faq-answer">We welcome walk-ins! For groups of 6+ or special occasions, call ahead and we&apos;ll do our best to accommodate.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">Do you have vegetarian/vegan options?</div>
              <div className="faq-answer">Yes! We always have several vegetarian options and can usually accommodate vegan requests. Just ask our staff.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">Is there parking?</div>
              <div className="faq-answer">Street parking is available, and there&apos;s a public lot just a 2-minute walk away. We&apos;re also easily accessible by public transit.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">Do you host events?</div>
              <div className="faq-answer">We can accommodate small private events. Call us to discuss your needs and we&apos;ll work something out.</div>
            </div>
          </div>
        </section>

        <section className="section section--band">
          <div className="section-header section-header--center">
            <h2>Come experience the difference</h2>
            <p>Join 500+ happy diners who make us their regular spot</p>
          </div>
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <p style={{ fontSize: '1.1rem', marginBottom: '1rem' }}>
              Visit us at <a href="tel:+34123456789" style={{ color: '#ffffff', fontWeight: '700' }}>[Street address]</a> or call <a href="tel:+34123456789" style={{ color: '#ffffff', fontWeight: '700' }}>+34 123 456 789</a>
            </p>
          </div>
        </section>

        <footer className="footer">
          <div>
            <h4>Find us</h4>
            <p>[Street address]</p>
            <p>5‑minute walk from [Landmark / station]</p>
            <p>Street parking available</p>
          </div>
          <div>
            <h4>Opening hours</h4>
            <p>Mon–Thu: 8:00–22:00</p>
            <p>Fri–Sat: 8:00–24:00</p>
            <p>Sun: 9:00–21:00</p>
          </div>
          <div>
            <h4>Contact</h4>
            <p>Phone: <a href="tel:+34123456789">+34 123 456 789</a></p>
            <p>Instagram: @cafexample</p>
            <p>Email: hello@cafexample.com</p>
          </div>
        </footer>
      </div>
    </div>
  );
}

function TutoringPage() {
  return (
    <div className="page page-tutoring">
      <div className="page-inner">
        <section className="hero">
          <div className="hero-layout">
            <div>
              <p className="hero-kicker">Tutoring &amp; language academy • [City]</p>
              <h1 className="hero-title">
                Supportive tutoring that focuses on confidence and results
              </h1>
              <p className="hero-subtitle">
                Small groups and one‑to‑one sessions for school subjects,
                languages and exam preparation, led by experienced teachers.
              </p>
              <p className="hero-meta">
                <span className="hero-pill">
                  <span>500+ students</span>
                  <span>95% improved grades within 3 months</span>
                </span>
              </p>
            </div>
            <aside className="hero-aside">
              <div className="hero-media">
                <img
                  className="hero-media-img"
                  src="https://images.unsplash.com/photo-1513258496099-48168024aec0?auto=format&fit=crop&w=900&q=80"
                  alt="Tutor helping a student at a desk"
                />
              </div>
              <div className="info-card">
                <h3>Who we help</h3>
                <ul className="list">
                  <li>Primary and secondary students needing steady support</li>
                  <li>Teens preparing for key exams</li>
                  <li>Adults learning languages for work or travel</li>
                  <li>Students with learning differences</li>
                </ul>
              </div>
            </aside>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>Programs at a glance</h2>
            <p>
              Comprehensive support across subjects and skill levels, tailored
              to each student&apos;s needs and learning style.
            </p>
          </header>
          <div className="grid grid-3">
            <div className="card card--highlight">
              <h3>Math &amp; science</h3>
              <p>From homework help to exam‑level problem solving.</p>
              <ul className="list">
                <li>Algebra, geometry and calculus</li>
                <li>Physics, chemistry and biology basics</li>
                <li>Test prep strategies</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Languages</h3>
              <p>Practical speaking and listening with solid grammar support.</p>
              <ul className="list">
                <li>Spanish, English, French and more</li>
                <li>Conversation‑focused sessions</li>
                <li>Exam preparation (Cambridge, IELTS)</li>
              </ul>
            </div>
            <div className="card card--highlight">
              <h3>Exam preparation</h3>
              <p>Structured plans for important school and language exams.</p>
              <ul className="list">
                <li>Local school exams</li>
                <li>Cambridge, IELTS and similar</li>
                <li>SAT and university entrance tests</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header section-header--center">
            <h2>Why students succeed with us</h2>
            <p>Our approach makes the difference</p>
          </header>
          <div className="grid grid-4">
            <div className="card card--feature">
              <div className="card--feature-icon">📚</div>
              <h3>Personalized plans</h3>
              <p>Every student gets a custom learning plan based on their goals and current level.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">👨‍🏫</div>
              <h3>Experienced teachers</h3>
              <p>All our tutors are certified with 5+ years of teaching experience.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">📊</div>
              <h3>Progress tracking</h3>
              <p>Regular assessments and updates keep students and parents informed of progress.</p>
            </div>
            <div className="card card--feature">
              <div className="card--feature-icon">💪</div>
              <h3>Confidence building</h3>
              <p>We focus on building confidence as much as improving grades.</p>
            </div>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>How our approach works</h2>
          </header>
          <div className="grid grid-2">
            <div>
              <p>
                We begin with a short conversation and a simple assessment to see
                where the student is now. From there, we build a custom plan and
                adjust it as they progress. Regular feedback keeps families in the
                loop and helps students stay motivated.
              </p>
              <div className="stats">
                <div className="stat">
                  <div className="stat-label">Session length</div>
                  <div className="stat-value">60–90 minutes</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Updates</div>
                  <div className="stat-value">Regular feedback to families</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Success rate</div>
                  <div className="stat-value">95% see improvement</div>
                </div>
              </div>
            </div>
            <div>
              <div className="testimonial">
                <p>
                  &ldquo;My son went from barely passing to feeling in control of
                  maths. The tutor explained things in a way that finally
                  clicked. His confidence has improved so much.&rdquo;
                </p>
                <span>— Maria G., parent</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;I reached the language level I needed for work faster
                  than I expected, and the classes were actually enjoyable.
                  Great teachers and flexible scheduling.&rdquo;
                </p>
                <span>— John P., [City]</span>
              </div>
              <div className="testimonial">
                <p>
                  &ldquo;The personalized approach made all the difference. My
                  daughter went from struggling to excelling in just one semester.
                  Highly recommend!&rdquo;
                </p>
                <span>— David R., parent</span>
              </div>
              <p className="hero-meta">
                <span className="pill-rating">
                  <strong>4.9/5</strong>
                  <span>Average rating from students &amp; parents</span>
                </span>
              </p>
            </div>
          </div>
        </section>

        <section className="section">
          <header className="section-header">
            <h2>Getting started</h2>
            <p>Simple steps to begin your learning journey</p>
          </header>
          <div className="process-container">
            <div className="process-step">
              <h3>Initial consultation</h3>
              <p>We meet with you (and parents if applicable) to understand goals, learning style, and schedule preferences.</p>
            </div>
            <div className="process-step">
              <h3>Assessment</h3>
              <p>A simple, low-pressure assessment helps us understand current level and identify areas for growth.</p>
            </div>
            <div className="process-step">
              <h3>Custom plan</h3>
              <p>We create a personalized learning plan with clear goals and milestones tailored to your needs.</p>
            </div>
            <div className="process-step">
              <h3>Regular sessions</h3>
              <p>Weekly sessions with progress updates. We adjust the plan as you improve and your goals evolve.</p>
            </div>
          </div>
        </section>

        <section className="section section--muted">
          <header className="section-header">
            <h2>Common questions</h2>
          </header>
          <div>
            <div className="faq-item">
              <div className="faq-question">What ages do you teach?</div>
              <div className="faq-answer">We work with students from primary school through adults. Our tutors are experienced with all age groups and learning styles.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">Do you offer online sessions?</div>
              <div className="faq-answer">Yes! We offer both in-person and online sessions. Many students find online just as effective, especially for language learning.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">How long until we see results?</div>
              <div className="faq-answer">Most students see improvement within 4-6 weeks. 95% show significant progress within 3 months of regular sessions.</div>
            </div>
            <div className="faq-item">
              <div className="faq-question">What if my child doesn&apos;t connect with a tutor?</div>
              <div className="faq-answer">No problem! We can switch tutors to find the right fit. The relationship between student and tutor is crucial for success.</div>
            </div>
          </div>
        </section>

        <section className="section section--band">
          <div className="section-header section-header--center">
            <h2>Ready to unlock your potential?</h2>
            <p>Join 500+ students who&apos;ve improved their grades and confidence with us</p>
          </div>
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <p style={{ fontSize: '1.1rem', marginBottom: '1rem' }}>
              Call <a href="tel:+34123456789" style={{ color: '#ffffff', fontWeight: '700' }}>+34 123 456 789</a> or visit us at [Street address]
            </p>
          </div>
        </section>

        <footer className="footer">
          <div>
            <h4>Where we teach</h4>
            <p>[Street address], [City]</p>
            <p>Online sessions also available</p>
            <p>Multiple locations across [City]</p>
          </div>
          <div>
            <h4>Typical schedule</h4>
            <p>Mon–Fri: 15:00–21:00</p>
            <p>Sat: 10:00–14:00</p>
            <p>Sun: Closed</p>
          </div>
          <div>
            <h4>Contact</h4>
            <p>Phone: <a href="tel:+34123456789">+34 123 456 789</a></p>
            <p>Email: hello@academyexample.com</p>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
